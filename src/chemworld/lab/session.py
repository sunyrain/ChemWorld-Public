"""Stateful, provider-free sessions for the public browser lab."""

from __future__ import annotations

import tempfile
import threading
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from typing import Any, cast

import gymnasium as gym

import chemworld  # noqa: F401
from chemworld.data.logging import TrajectoryLogger, load_jsonl, observation_to_json, to_builtin
from chemworld.envs.chemworld_env import ChemWorldEnv
from chemworld.eval.verify import verify_records
from chemworld.lab.http_security import RequestRejected
from chemworld.materials import action_material_display
from chemworld.tasks import get_task, list_tasks

_TASK_COPY: dict[str, tuple[str, str, str]] = {
    "reaction-to-assay": (
        "投料到最终检测",
        "完成投料、反应控制、终止与 final assay，学习最短的完整实验闭环。",
        "在有限预算中得到有效终点分数，同时保持轨迹合法。",
    ),
    "reaction-to-purification": (
        "反应与纯化",
        "把反应和两相分离串成一个可回放流程，观察纯度、回收率与成本的权衡。",
        "获得更纯的目标产物，同时控制转移损失。",
    ),
    "reaction-to-crystallization": (
        "反应与冷却结晶",
        "从反应生成目标物，再通过加晶种、冷却和过滤回收晶体。",
        "平衡晶体收率、纯度、粒度与过程预算。",
    ),
    "reaction-to-distillation": (
        "反应与蒸馏切割",
        "利用挥发性差异选择蒸馏条件和馏分切割。",
        "在纯度、回收率和溶剂损失之间作出可解释选择。",
    ),
    "flow-reaction-optimization": (
        "连续流反应优化",
        "通过流量、停留时间与温度控制连续流转化。",
        "兼顾转化、吞吐量和安全风险。",
    ),
    "electrochemical-conversion": (
        "电化学转化",
        "使用电位、电流和反应时间控制虚拟电化学转化。",
        "提高选择性和能量效率，同时避免过强驱动力。",
    ),
}

_EFFECTS: dict[str, tuple[str, str, str]] = {
    "configure_instrument": (
        "配置光谱采集",
        "设置扫描次数、分辨率和稀释倍率；measure 才采集。",
        "measure",
    ),
    "set_flow_stream": ("设置进出料流", "设置声明连接的阀流量；零值关闭阀门，不推进时间。", "flow"),
    "advance_flow": (
        "推进有限库存连续流",
        "实际扣减上游库存并收集出口；同时记账热量与电荷。",
        "flow",
    ),
    "configure_control": ("配置过程控制", "配置持久过程控制；不会自动推进时间。", "heat"),
    "queue_control_stage": ("追加控制阶段", "向有限队列追加声明的设定值阶段。", "heat"),
    "set_control_feedback": ("设定传感反馈", "设定温度或压力阈值及合同允许的响应。", "measure"),
    "advance_control": ("推进控制程序", "按实际时间、能量及气体收支记账。", "heat"),
    "pause_control": ("暂停程序", "暂停持久控制；不是完成终检。", "terminate"),
    "resume_control": ("恢复程序", "从持久游标恢复，不重做已完成阶段。", "heat"),
    "select_vessel": ("切换容器", "切换到声明容器，消耗操作预算但不推进物理时间。", "transfer"),
    "route_material": ("沿端口转移", "沿声明连接转移物料，遵守显式混合政策。", "transfer"),
    "add_component": ("独立组分投料", "加入公开投料目录的单个组分，按实际摩尔数记账。", "feed"),
    "create_container": ("创建容器", "创建有身份的空容器，不产生物料。", "transfer"),
    "transfer_material": ("路由物料", "按公开合同转移有身份的物料，保留来源与损失。", "transfer"),
    "resuspend_crystals": ("复悬晶体", "将隔离晶体放回液体处理状态。", "crystallize"),
    "add_solvent": ("累计加液", "向当前容器加入溶剂并增加体积。", "feed"),
    "add_reagent": ("累计投料", "向当前容器加入试剂，已有物料不会被替换。", "feed"),
    "add_catalyst": ("加入催化剂", "向当前容器加入催化剂。", "feed"),
    "heat": ("加热反应", "从当前组成继续演化并累计时间与热历程。", "heat"),
    "wait": ("继续反应", "当前组成继续随时间演化。", "mix"),
    "sample": ("取样", "从当前库存按比例扣除样品。", "sample"),
    "measure": ("仪器测量", "消耗样品和成本并返回公开观测。", "measure"),
    "quench": ("淬灭", "停止反应并把容器切换到淬灭状态。", "quench"),
    "terminate": ("终止实验", "标记当前过程已终止；final assay 才会完成实验。", "terminate"),
    "add_phase": ("加入液相", "向当前相账本累计加入液体。", "phase"),
    "add_extractant": ("加入萃取相", "向当前容器加入萃取相。", "phase"),
    "mix": ("混合传质", "重新分配相库存并累计混合时间。", "mix"),
    "settle": ("静置分层", "累计静置时间并更新分层状态。", "settle"),
    "separate_phase": ("选择工作相", "选择一个现有相用于后续操作。", "separate"),
    "wash": ("洗涤", "加入洗液并重新分配所选相。", "wash"),
    "dry": ("干燥", "移除水分并记录液体损失。", "dry"),
    "concentrate": ("浓缩", "减少液体库存并累计时间与损失。", "evaporate"),
    "transfer": ("转移", "转移库存并计入釜底和管线残留。", "transfer"),
    "seed_crystals": ("加入晶种", "向当前结晶实验加入晶种。", "seed"),
    "cool_crystallize": ("冷却结晶", "更新固液状态并累计冷却时间。", "crystallize"),
    "filter_crystals": ("过滤晶体", "转移固相并计入回收损失。", "filter"),
    "evaporate": ("蒸发", "移除部分液体并累计热负荷。", "evaporate"),
    "distill": ("蒸馏", "把库存拆分为馏出相与釜底相。", "distill"),
    "collect_fraction": ("收集馏分", "转移部分馏出相并计入损失。", "collect"),
    "set_flow_rate": ("设定流量", "更新流量与停留时间，本动作不处理物料。", "flow"),
    "run_flow": ("运行连续流", "按当前配置处理库存。", "flow"),
    "set_potential": ("设定电位", "更新电化学配置，本动作不发生电解。", "electro"),
    "electrolyze": ("运行电解", "按当前配置继续处理组成。", "electro"),
}


def _title(task_id: str) -> str:
    return task_id.replace("-", " ").title()


def task_catalog() -> list[dict[str, Any]]:
    """Return public task cards consumed by the browser application."""

    cards: list[dict[str, Any]] = []
    for task in list_tasks():
        title, background, goal = _TASK_COPY.get(
            task.task_id,
            (_title(task.task_id), task.description, "设计合法、可回放且资源受控的实验路径。"),
        )
        cards.append(
            {
                "task_id": task.task_id,
                "title": title,
                "background": background,
                "student_goal": goal,
                "description": task.description,
                "budget": task.budget,
                "episode_mode": task.episode_mode,
                "seeds": list(task.seeds),
                "success_metrics": list(task.success_metrics),
                "physics_maturity": task.kernel_maturity.lowest_level.value,
                "proxy_allowed": task.kernel_maturity.proxy_allowed,
            }
        )
    cards.sort(key=lambda item: (item["task_id"] != "reaction-to-assay", item["task_id"]))
    return cards


def _effect(operation: str) -> dict[str, str]:
    label, summary, visual = _EFFECTS.get(
        operation,
        ("状态更新", "更新当前公开实验状态。", "generic"),
    )
    return {"label": label, "summary": summary, "visual": visual}


def _affordance(entry: dict[str, Any]) -> dict[str, Any]:
    schema = dict(entry.get("schema") or entry)
    operation = str(entry.get("operation") or schema.get("operation") or "")
    return {
        "operation": operation,
        "required_fields": list(schema.get("required_fields") or []),
        "fields": deepcopy(schema.get("fields") or []),
        "preconditions": list(schema.get("preconditions") or []),
        "effect": _effect(operation),
    }


def _state_effects(action: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    delta = dict(info.get("state_delta_summary") or {})
    return {
        **_effect(str(action.get("operation") or "")),
        "delta_time_s": float(delta.get("delta_time_s", 0.0)),
        "delta_cost": float(delta.get("delta_cost", info.get("cost_delta", 0.0))),
        "delta_risk": float(delta.get("delta_risk", info.get("risk_delta", 0.0))),
        "delta_temperature_K": float(delta.get("delta_temperature_K", 0.0)),
        "delta_volume_L": float(delta.get("delta_volume_L", 0.0)),
        "sample_delta_L": float(info.get("sample_delta", 0.0)),
        "transaction_status": str(info.get("transaction_status") or "unknown"),
    }


@dataclass
class LabSession:
    """One manual browser session backed by the normal public Gym environment."""

    task_id: str
    seed: int
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        task = get_task(self.task_id)
        self._env = gym.make("ChemWorld", **task.env_kwargs(seed=self.seed))
        self._env.reset(seed=self.seed)
        base = cast(ChemWorldEnv, self._env.unwrapped)
        self._task_info = {**base.task_info(), **base.evaluator_provenance()}
        self._history: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self._closed = False
        self._storage = tempfile.TemporaryDirectory(prefix="chemworld-lab-")
        self._trajectory = Path(self._storage.name) / "trajectory.jsonl"
        self._logger = TrajectoryLogger(self._trajectory)
        self._replay: dict[str, Any] | None = None

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._logger.close()
            self._env.close()
            self._storage.cleanup()

    def export(self) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise ValueError("Session is closed")
            return {
                "trajectory_jsonl": self._trajectory.read_text(encoding="utf-8"),
                "history": list(self._history),
                "task_id": self.task_id,
                "seed": self.seed,
            }

    def replay(self) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise ValueError("Session is closed")
            if not self._history:
                raise ValueError("No executed actions to replay")
            if self._replay is None:
                self._replay = verify_records(load_jsonl(self._trajectory), tolerance=0.0).to_dict()
            return dict(self._replay)

    def state(self) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise ValueError("Session is closed")
            base: Any = self._env.unwrapped
            campaign = to_builtin(base.campaign_state())
            report = to_builtin(base.observation_view("lab_report"))
            experiment_index = int(campaign.get("experiment_index", 0))
            current = [
                row
                for row in self._history
                if int(row.get("experiment_index", -1)) == experiment_index
            ]
            effects = [dict(row.get("state_effects") or {}) for row in current]
            return {
                "session_id": self.session_id,
                "task_id": self.task_id,
                "seed": self.seed,
                "campaign_state": campaign,
                "lab_report": report,
                "available_actions": [_affordance(item) for item in base.available_actions()],
                "history": list(self._history),
                "public_vessel": {
                    "experiment_index": experiment_index,
                    "vessel_relation": "cumulative" if current else "fresh",
                    "operation_count": len(current),
                    "net_volume_delta_L": sum(
                        float(item.get("delta_volume_L", 0.0)) for item in effects
                    ),
                    "elapsed_time_delta_s": sum(
                        float(item.get("delta_time_s", 0.0)) for item in effects
                    ),
                    "sampled_volume_L": sum(
                        float(item.get("sample_delta_L", 0.0)) for item in effects
                    ),
                },
                "done": bool(campaign.get("done")),
            }

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise ValueError("Session is closed")
            if len(self._history) >= 1000:
                raise ValueError("Session action limit reached; export and close this session")
            base: Any = self._env.unwrapped
            validation = dict(base.validate_action(action))
            if not validation.get("valid", False):
                available = [item["operation"] for item in base.available_actions()]
                return {
                    "accepted": False,
                    "validation": to_builtin(validation),
                    "feedback": {
                        "message": "动作未执行，实验状态和预算均未改变。",
                        "recovery_suggestion": "当前合法动作：" + "、".join(available[:8]),
                    },
                    "state": self.state(),
                }
            campaign_before = base.campaign_state()
            observation, reward, terminated, truncated, info = self._env.step(action)
            self._replay = None
            self._logger.log(
                task_info=self._task_info,
                step=len(self._history) + 1,
                action=action,
                observation=observation,
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                info=info,
                agent_metadata={"agent_id": "manual-lab"},
            )
            report = to_builtin(base.observation_view("lab_report"))
            campaign = to_builtin(base.campaign_state())
            canonical = dict(validation.get("canonical_action") or action)
            record = {
                "step": len(self._history) + 1,
                "experiment_index": int(campaign_before.get("experiment_index", 0)),
                "action": to_builtin(canonical),
                "action_display": action_material_display(canonical),
                "reward": float(reward),
                "leaderboard_score": to_builtin(info.get("leaderboard_score")),
                "best_score": campaign.get("best_score"),
                "visible_metrics": report.get("visible_metrics", {}),
                "constraint_flags": to_builtin(info.get("constraint_flags", {})),
                "observation": observation_to_json(observation),
                "status": report.get("status"),
                "state_effects": _state_effects(canonical, info),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
            self._history.append(record)
            return {
                "accepted": info["transaction_status"] == "committed",
                "validation": to_builtin(validation),
                "feedback": {
                    "message": report.get("text"),
                    "visible_metrics": report.get("visible_metrics", {}),
                    "recovery_suggestion": report.get("recovery_suggestion"),
                },
                "record": record,
                "state": self.state(),
            }


class LabSessionManager:
    """Thread-safe in-memory session registry."""

    def __init__(self, max_sessions: int = 16) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be positive")
        self.max_sessions = max_sessions
        self._sessions: dict[str, LabSession] = {}
        self._lock = threading.RLock()

    def create(self, task_id: str, seed: int | None = None) -> LabSession:
        task = get_task(task_id)
        with self._lock:
            if len(self._sessions) >= self.max_sessions:
                raise RequestRejected(
                    "Close an existing session first", HTTPStatus.SERVICE_UNAVAILABLE
                )
            session = LabSession(task_id=task_id, seed=task.seeds[0] if seed is None else seed)
            self._sessions[session.session_id] = session
        return session

    def close(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self.get(session_id)
            with session._lock:
                exported = session.export()
                count = session.state()["campaign_state"]["final_assay_count"]
                session.close()
                del self._sessions[session_id]
            return {"closed": True, "final_assay_count": count, "export": exported}

    def get(self, session_id: str) -> LabSession:
        with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError as exc:
                raise KeyError(f"unknown lab session: {session_id}") from exc

    def close_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.close()


__all__ = ["LabSession", "LabSessionManager", "task_catalog"]
