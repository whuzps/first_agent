from datetime import datetime
import logging

logger = logging.getLogger(__name__)
import os
import traceback

from langgraph.errors import GraphRecursionError
from langgraph.types import Command
from core.config import MAX_ATTEMPTS, LOG_DIR
import asyncio

from core.state import StateStatus

class Robust:
    def __init__(self, graph_obj, checkpointer):
        self.graph = graph_obj
        self.checkpointer = checkpointer
    
    async def run_with_recovery(
        self, 
        input_data: dict, 
        thread_id: str,
        max_recovery_attempts: int = MAX_ATTEMPTS
    ):
        """带恢复机制的图执行"""
        attempt = 0
        last_error = None
        
        while attempt < max_recovery_attempts:
            try:
                config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "recursion_limit": 100
                    }
                }

                # ── 前置清理：若 checkpoint 中有残留的 HITL 中断状态，自动 reject ──
                await self._clear_stale_interrupt(config, thread_id)

                result = await self.graph.ainvoke(input_data, config)

                # ── HITL 中断检测：检查图是否因 interrupt() 暂停 ──
                state_snapshot = await self.graph.aget_state(config)
                if state_snapshot and state_snapshot.next:
                    # 图被中断（有待执行的节点），提取中断信息
                    interrupt_info = None
                    if hasattr(state_snapshot, "tasks"):
                        for task in state_snapshot.tasks:
                            if hasattr(task, "interrupts") and task.interrupts:
                                interrupt_info = task.interrupts[0].value
                                break
                    logger.info(
                        f"HITL 中断检测：图被中断，pending_nodes={state_snapshot.next}，"
                        f"interrupt_info={interrupt_info}"
                    )
                    result["__interrupted__"] = True
                    result["__interrupt_info__"] = interrupt_info
                    result["__thread_id__"] = thread_id
                return result
                
            except GraphRecursionError as e:
                await self.handle_recursion_error(thread_id, e)
                break
                
            except Exception as e:
                last_error = e
                attempt += 1
                
                await self.log_error(thread_id, "全局重试", e, attempt)
                
                if attempt < max_recovery_attempts:
                    await self.recover_from_checkpoint(thread_id)
                    await asyncio.sleep(2 ** attempt)
        
        await self.send_to_dead_letter(thread_id, input_data, last_error)
        raise last_error

    async def _clear_stale_interrupt(self, config: dict, thread_id: str) -> None:
        """检查 checkpoint 中是否有残留的 HITL 中断，自动 reject 以恢复图到可用状态。

        场景：Redis HITL key 已过期（5min TTL），但 PostgreSQL checkpoint 中图仍处于
        interrupted 状态。此时新的 ainvoke 会打到一个中断态的图上导致异常行为。
        """
        try:
            state_snapshot = await self.graph.aget_state(config)
            if state_snapshot and state_snapshot.next:
                logger.warning(
                    f"[Robust] 检测到 checkpoint 中有残留中断状态: thread_id={thread_id}，"
                    f"pending_nodes={state_snapshot.next}，执行自动 reject"
                )
                await self.graph.ainvoke(Command(resume="rejected"), config)
                logger.info(f"[Robust] 残留中断已自动 reject: thread_id={thread_id}")
        except Exception as e:
            logger.warning(
                f"[Robust] 清理残留中断失败（将继续正常执行）: thread_id={thread_id}, error={e}"
            )

    async def resume_from_interrupt(
        self,
        thread_id: str,
        user_decision: str,
    ) -> dict:
        """从 HITL 中断处恢复图执行。

        Args:
            thread_id: 会话线程 ID（必须与被中断的请求一致）
            user_decision: 用户决定（"approved" 或 "rejected"）
        """
        config = {
            "configurable": {
                "thread_id": thread_id,
                "recursion_limit": 100
            }
        }
        logger.info(f"HITL 恢复执行：thread_id={thread_id}，user_decision={user_decision}")

        result = await self.graph.ainvoke(
            Command(resume=user_decision),
            config,
        )
        return result
    
    async def recover_from_checkpoint(self, thread_id: str):
        """从最后一个成功的checkpoint恢复"""
        # 获取最后一个成功的状态
        config = {"configurable": {"thread_id": thread_id}}
        
        # 兼容不同 checkpointer 实现（如 AsyncPostgresSaver）
        if hasattr(self.checkpointer, "alist"):
            checkpoints = []
            async for c in self.checkpointer.alist(config, limit=10):
                checkpoints.append(c)
        else:
            checkpoints = list(self.checkpointer.list(config, limit=10))
        
        for checkpoint in checkpoints:
            if checkpoint.metadata.get("status") == StateStatus.SUCCESS:
                # 恢复到这个状态
                if hasattr(self.checkpointer, "aput"):
                    await self.checkpointer.aput(
                        config,
                        checkpoint.checkpoint,
                        checkpoint.metadata,
                        {}
                    )
                else:
                    self.checkpointer.put(
                        config,
                        checkpoint.checkpoint,
                        checkpoint.metadata,
                        {}
                    )
                break

    async def log_error(self, thread_id: str, content: str, e: Exception, attempt: int = None):
        """记录报错日志"""
        logger.error(f"Thread_id {thread_id}, content: {content}, attempt {attempt}, error: {e}")
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
            line = f"{ts} | ERROR | thread_id {thread_id}, content: {content}, retry attempt {attempt}, error: {e} "
            with open(LOG_DIR / "error.log", "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            # 打印错误堆栈
            logger.error(traceback.format_exc())
            pass

    async def handle_recursion_error(self, thread_id: str, e: Exception):
        logger.error(f"recursion error. thread_id {thread_id}, error: {e}")
        """递归深度超限"""
        pass

    async def send_to_dead_letter(self, thread_id: str, query: str, last_error: Exception = None):
        """将失败的请求发送到死信队列（这里简单写入文件）"""
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            import json
            record = {
                "timestamp": ts,
                "thread_id": thread_id,
                "query": str(query),
                "error": str(last_error)
            }
            
            with open(os.path.join(LOG_DIR, "dead_letter.json"), "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
            logger.info(f"已将失败请求写入死信队列: {thread_id}")
        except Exception as e:
            logger.error(f"写入死信队列失败: {e}")