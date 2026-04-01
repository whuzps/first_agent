import json
import uuid
import random
import time
from locust import HttpUser, task, between, events

class ChatUser(HttpUser):
    # 用户请求之间的等待时间，模拟真实用户的思考时间 (1到5秒)
    wait_time = between(1, 5)

    def on_start(self):
        # 每个虚拟用户启动时生成唯一的 user_id 和 thread_id
        self.user_id = f"locust_user_{uuid.uuid4().hex[:8]}"
        self.thread_id = f"locust_thread_{uuid.uuid4().hex[:8]}"
        self.headers = {
            "Content-Type": "application/json",
            "X-Tenant-ID": "default"
        }

    @task(3)
    def chat_stream_greeting(self):
        """模拟常见的问候语或简单查询"""
        queries = ["你好", "在吗", "人工服务", "你们有哪些产品？", "我想查一下订单"]
        self._send_chat_request(random.choice(queries))

    @task(1)
    def chat_stream_complex(self):
        """模拟较复杂的业务查询"""
        queries = [
            "我的订单ORD-1234物流状态怎么样了？", 
            "产品有什么售后保障？", 
            "怎么申请退款？", 
            "可以开发票吗？"
        ]
        self._send_chat_request(random.choice(queries))

    def _send_chat_request(self, query):
        payload = {
            "query": query,
            "user_id": self.user_id,
            "thread_id": self.thread_id
        }

        # 记录请求开始时间
        start_time = time.time()
        
        # 发送 POST 请求，由于是 SSE 接口，设置 stream=True
        with self.client.post(
            "/chat/stream", 
            json=payload, 
            headers=self.headers, 
            stream=True, 
            catch_response=True,
            name="/chat/stream"
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP Status {response.status_code}: {response.text}")
                return

            first_token_received = False
            full_response = ""
            error_occurred = False

            try:
                # 遍历 SSE 流的每一行数据
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data: "):
                            data_str = decoded_line[6:] # 去除 "data: " 前缀
                            try:
                                data = json.loads(data_str)
                                event_type = data.get("type")
                                
                                if event_type == "token":
                                    if not first_token_received:
                                        first_token_received = True
                                        # 可以在这里记录 TTFT (Time To First Token) 如果需要的话
                                        ttft = time.time() - start_time
                                        events.request.fire(
                                            request_type="SSE",
                                            name="/chat/stream TTFT",
                                            response_time=ttft * 1000,
                                            response_length=0,
                                            exception=None,
                                            context={}
                                        )
                                    full_response += data.get("content", "")
                                
                                elif event_type == "error":
                                    error_occurred = True
                                    response.failure(f"Stream Error: {data.get('message')}")
                                    break
                                
                                elif event_type == "done":
                                    # 正常结束
                                    break
                                
                            except json.JSONDecodeError:
                                # 可能遇到非 JSON 格式的行
                                pass
                
                if not error_occurred:
                    response.success()
                    
            except Exception as e:
                response.failure(f"Exception during stream parsing: {str(e)}")

if __name__ == "__main__":
    # 使用说明
    print("运行此脚本: locust -f locust_chat.py --host=http://localhost:8000")
