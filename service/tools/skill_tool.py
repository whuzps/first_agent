"""渐进式披露（Progressive Disclosure）技能工具

三层架构：
  L0 索引层 — SkillRegistry 自动扫描 skills/ 目录，提取 frontmatter 构建轻量索引（name + description），
              注入 System Prompt，LLM 知道有哪些技能可用但不占用过多 token。
  L1 详情层 — lookup_skill 工具，LLM 按需调用，加载完整 SKILL.md 操作指南。
  L2 引用层 — read_reference 工具，LLM 需要深入了解时调用，加载技能内的参考文档/脚本。
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Skills 根目录：service/skills/
_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


class SkillRegistry:
    """技能注册表（单例）：扫描 skills/ 目录，解析 SKILL.md frontmatter 构建轻量索引。

    索引结构：{ skill_name: { name, description, dir, skill_md } }
    """

    _instance: Optional["SkillRegistry"] = None

    def __init__(self):
        self._index: Dict[str, dict] = {}
        self._scan()

    @classmethod
    def get(cls) -> "SkillRegistry":
        """获取全局单例（懒加载）"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """重置单例（用于测试或热更新技能目录后重新扫描）"""
        cls._instance = None

    # ── 内部方法 ──────────────────────────────────────────────────

    @staticmethod
    def _parse_frontmatter(content: str) -> Dict[str, str]:
        """解析 YAML frontmatter（--- 包裹的头部元数据）"""
        match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        if not match:
            return {}
        try:
            return yaml.safe_load(match.group(1)) or {}
        except Exception:
            return {}

    def _scan(self):
        """扫描 skills/ 目录，为每个含有 SKILL.md 且 frontmatter 非空的子目录建立索引"""
        if not _SKILLS_DIR.is_dir():
            logger.warning(f"技能目录不存在: {_SKILLS_DIR}")
            return

        for skill_dir in sorted(_SKILLS_DIR.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue

            content = skill_md.read_text(encoding="utf-8")
            meta = self._parse_frontmatter(content)
            name = meta.get("name", skill_dir.name)
            description = meta.get("description", "")

            # 没有描述的技能不纳入索引（视为未完成的占位）
            if not description.strip():
                logger.debug(f"跳过无描述的技能目录: {skill_dir.name}")
                continue

            self._index[name] = {
                "name": name,
                "description": description,
                "dir": skill_dir,
                "skill_md": skill_md,
            }
            logger.info(f"已注册技能: {name} — {description[:80]}...")

        logger.info(f"技能注册完成，共 {len(self._index)} 个技能")

    # ── 公开 API ──────────────────────────────────────────────────

    def get_index_summary(self) -> str:
        """返回所有已注册技能的名称+描述摘要（用于注入 System Prompt，L0 层）"""
        if not self._index:
            return "（暂无可用技能）"
        lines = []
        for name, info in self._index.items():
            lines.append(f"- {name}: {info['description']}")
        return "\n".join(lines)

    def get_skill_body(self, skill_name: str) -> Optional[str]:
        """读取指定技能的完整 SKILL.md 内容（去掉 frontmatter，L1 层）"""
        info = self._index.get(skill_name)
        if not info:
            return None
        content = info["skill_md"].read_text(encoding="utf-8")
        return re.sub(r"^---\n.*?\n---\n?", "", content, count=1, flags=re.DOTALL).strip()

    def get_reference_path(self, skill_name: str, reference: str) -> Optional[Path]:
        """获取技能内部引用文件的安全路径（L2 层，防止路径穿越攻击）"""
        info = self._index.get(skill_name)
        if not info:
            return None
        target = (info["dir"] / reference).resolve()
        # 安全校验：目标路径必须在技能目录内
        if not str(target).startswith(str(info["dir"].resolve())):
            logger.warning(f"路径穿越检测: skill={skill_name}, ref={reference}")
            return None
        if not target.exists():
            return None
        return target

    def list_references(self, skill_name: str) -> List[str]:
        """列出技能目录下所有可引用的附属文件（排除 SKILL.md）"""
        info = self._index.get(skill_name)
        if not info:
            return []
        return [
            str(f.relative_to(info["dir"]))
            for f in sorted(info["dir"].rglob("*"))
            if f.is_file() and f.name != "SKILL.md"
        ]

    def skill_names(self) -> List[str]:
        """返回所有已注册技能的名称列表"""
        return list(self._index.keys())


# ═══════════════════════════════════════════════════════════════
#  LangChain Tools — 供 ReAct 节点按需调用（渐进式披露 L1 / L2）
# ═══════════════════════════════════════════════════════════════


@tool
def lookup_skill(skill_name: str) -> str:
    """查找并读取指定技能的完整操作指南（L1 渐进式披露）。

    当你需要了解某个技能的详细操作流程、可用工具列表、注意事项和示例场景时，
    调用此工具加载完整指南。技能名称可从系统提示中的【可用技能索引】获取。

    Args:
        skill_name: 技能名称，如 order-management
    """
    registry = SkillRegistry.get()
    body = registry.get_skill_body(skill_name)

    if body is None:
        available = registry.skill_names()
        return f"未找到技能 '{skill_name}'。当前可用技能: {available}"

    # 如果技能目录下有附属文件，在尾部追加提示
    refs = registry.list_references(skill_name)
    if refs:
        body += "\n\n---\n📎 此技能还包含以下参考文档（如需查看可调用 read_reference）：\n"
        for r in refs:
            body += f"  - {r}\n"

    logger.info(f"lookup_skill: 已加载技能 '{skill_name}' 的完整指南")
    return body


@tool
def read_reference(skill_name: str, reference: str) -> str:
    """读取技能内部引用的参考文档或脚本（L2 渐进式披露）。

    当技能指南中提到参考文档（如错误码对照表、校验脚本等），
    调用此工具获取详细内容。

    Args:
        skill_name: 技能名称，如 order-management
        reference: 引用文件的相对路径，如 references/error-codes.md
    """
    registry = SkillRegistry.get()
    target = registry.get_reference_path(skill_name, reference)

    if target is None:
        refs = registry.list_references(skill_name)
        return f"未找到引用 '{reference}'（技能: {skill_name}）。可用引用: {refs}"

    content = target.read_text(encoding="utf-8")
    logger.info(f"read_reference: 已读取 {skill_name}/{reference}")
    return content
