import requests
import json
import operator
from typing import Annotated, Dict, Any, Literal, Optional
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field, field_validator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ==========================================
# 1. 定义状态 (State)
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list, operator.add] 
    current_params: Dict[str, Any] 
    final_result: Dict[str, Any]
    # 新增：用于在节点间传递意图
    intent: Literal["NEW_QUERY", "REFINE_QUERY"]

# ==========================================
# 2. 定义 Pydantic 结构
# ==========================================

# --- A. 专门用于意图分类的简单结构 ---
class IntentOutput(BaseModel):
    intent: Literal["NEW_QUERY", "REFINE_QUERY"] = Field(
        description="""
        用户意图分类：
        - NEW_QUERY: 全新查询。用户发起一个完全不同的任务，或者明确表示'不查那个了'、'重新查'。
        - REFINE_QUERY: 追加/修改。用户在当前基础上增加条件（如'只要超时的'）或修改条件（如'改查A站点的'）。
        """
    )

# --- B. 专门用于参数提取的结构 (回归纯粹) ---
# 不需要再去定义 search_intent 字段了，专心定义业务字段
class TaskSearchInput(BaseModel):
    # --- 枚举/ID/状态 ---
    task_type: Optional[int] = Field(default=None, description="精确匹配任务类型：1=临时任务, 2=固定任务, 3=空箱任务。")
    status: Optional[int] = Field(default=None, description="精确匹配任务状态：1=已创建, 2=运送中, 3=已到达, 4=已取消。")
    task_id: Optional[int] = Field(default=None, description="精确任务ID。")
    is_time_out: Optional[bool] = Field(default=None, description="是否超时 (True/False)。")
    is_active: Optional[bool] = Field(default=None, description="是否激活 (True/False)。")

    # --- 名称/条码/摘要 ---
    box_bar_code: Optional[str] = Field(default=None, description="箱子条码，支持模糊包含匹配。")
    source_station_name: Optional[str] = Field(default=None, description="源站点名称，支持模糊匹配。")
    dest_station_name: Optional[str] = Field(default=None, description="目标站点名称，支持模糊匹配。")
    summary: Optional[str] = Field(default=None, description="任务摘要/备注，支持模糊匹配。")
    source_station_id: Optional[int] = Field(default=None, description="源站点ID。")
    dest_station_id: Optional[int] = Field(default=None, description="目标站点ID。")

    # --- 时间范围 ---
    create_date_from: Optional[str] = Field(default=None, description="创建时间范围-起始。")
    create_date_to: Optional[str] = Field(default=None, description="创建时间范围-结束。")
    task_start_time_from: Optional[str] = Field(default=None, description="任务实际开始时间-起始。")
    task_start_time_to: Optional[str] = Field(default=None, description="任务实际开始时间-结束。")
    task_finish_time_from: Optional[str] = Field(default=None, description="任务完成时间-起始。")
    task_finish_time_to: Optional[str] = Field(default=None, description="任务完成时间-结束。")

    # --- 数值范围 ---
    min_consume_time: Optional[int] = Field(default=None, description="最小消耗时间(秒)。")
    max_consume_time: Optional[int] = Field(default=None, description="最大消耗时间(秒)。")
    min_estimated_time: Optional[int] = Field(default=None, description="最小预估时间(秒)。")
    max_estimated_time: Optional[int] = Field(default=None, description="最大预估时间(秒)。")

    # --- 分页 ---
    page: int = Field(default=1, description="页码，默认为1。")
    size: int = Field(default=10, description="每页大小，默认为10。")

    # =========================================================
    # 【修复核心】升级版清洗器：自动处理 List 和 Dict
    # =========================================================
    @field_validator('*', mode='before')
    @classmethod
    def preprocess_input(cls, v: Any) -> Any:
        # 1. 处理列表情况 (解决你现在的报错)
        if isinstance(v, list):
            # 如果是空列表 [] -> 返回 None
            if not v:
                return None
            # 如果是 ['神经内科'] -> 取出 '神经内科'
            return v[0]

        # 2. 处理字典情况 (解决之前的 {'value': 1} 问题)
        if isinstance(v, dict):
            if 'value' in v:
                return v['value']
            if not v:
                return None
            return None
            
        # 3. 正常值直接返回
        return v

# ==========================================
# 3. 定义节点 (Nodes) - 职责分离
# ==========================================

# --- 节点 1: 意图分类器 (Router) ---
def intent_classifier_node(state: AgentState):
    """
    只做一件事：判断是新查询还是老查询。
    """
    llm = ChatOpenAI(
        model="/model/DeepSeek-R1",
        openai_api_base="https://aimpapi.midea.com/t-aigc/aimp-deepseek-r1/v1",
        openai_api_key="msk-ca04571203246b31eec2dae635521ea079ca23818fdbe1f2177e17934382d378",
        temperature=0.01
        # extra_body={"chat_template_kwargs": {"enable_thinking": True}}
    )
    
    # 绑定简单的 IntentOutput
    classifier = llm.with_structured_output(IntentOutput)
    
    last_user_msg = state["messages"][-1].content
    
    # 极简 Prompt，没有任何参数干扰
    prompt = f"""
    请分析用户的最新指令："{last_user_msg}"
    
    判断他是想在【现有查询基础上修改条件】，还是【发起一个全新的查询】？
    如果话题完全切换（例如从'查任务'变成'查站点'，或者从'查神经内科'变成'查所有任务'），请判定为 NEW_QUERY。
    """
    
    try:
        result = classifier.invoke([SystemMessage(content=prompt)])
        intent = result.intent
    except:
        intent = "REFINE_QUERY" # 默认保守策略
        
    print(f"\n--------- 🧠 意图判断: {intent} ---------")
    return {"intent": intent}


# --- 节点 2: 状态清洗器 (Cleaner) ---
def state_cleaner_node(state: AgentState):
    """
    只做一件事：清空 current_params，重置为默认值。
    """
    print("   -> [动作] 检测到新话题，正在清除历史参数...")
    return {
        "current_params": {"page": 1, "size": 10} # 恢复出厂设置
    }


# --- 节点 3: 参数提取器 (Extractor) ---
def extract_params_node(state: AgentState):
    """
    只做一件事：提取参数并合并。
    它根本不用管'是不是新查询'，因为它拿到的 current_params 已经被上游节点处理过了。
    如果是新查询，它拿到的就是空的，自然就只提取新参数。
    """
    llm = ChatOpenAI(
        model="/model/DeepSeek-R1",
        openai_api_base="https://aimpapi.midea.com/t-aigc/aimp-deepseek-r1/v1",
        openai_api_key="msk-ca04571203246b31eec2dae635521ea079ca23818fdbe1f2177e17934382d378",
        temperature=0.01
        # extra_body={"chat_template_kwargs": {"enable_thinking": True}}
    )
    
    structured_llm = llm.with_structured_output(TaskSearchInput)
    
    current_p = state.get("current_params", {})
    # 兜底：如果前序节点没初始化，这里初始化
    if not current_p: current_p = {"page": 1, "size": 10}

    last_msg = state["messages"][-1].content
    
    # Prompt 只需要关注提取
    system_prompt = f"""
    当前生效参数：{json.dumps(current_p, ensure_ascii=False)}
    
    用户指令："{last_msg}"
    
    请输出需要变更的参数（增量）。
    注意：
    1. 提取明确提到的条件。
    2. 如果用户说“去掉/不限”，请输出 null。
    """
    
    try:
        res = structured_llm.invoke([SystemMessage(content=system_prompt)])
        # exclude_unset=True 依然重要，用于处理清除逻辑
        delta = res.model_dump(exclude_unset=True)
        
        print(f"[Debug] 提取增量: {delta}")
        
        # 合并
        merged = current_p.copy()
        for k, v in delta.items():
            if v is None:
                if k in merged: del merged[k]
            else:
                merged[k] = v
        
        # 再次兜底分页
        if "page" not in merged: merged["page"] = 1
        if "size" not in merged: merged["size"] = 10
        
        return {"current_params": merged}
        
    except Exception as e:
        print(f"[Error] 提取失败: {e}")
        return {"current_params": current_p}


# --- 节点 4: 数据请求器 (Fetcher) ---
def fetch_data_node(state: AgentState):
    params = state["current_params"]
    print(f"  >>> [API执行] 参数: {params}")
    
    # 简化的请求逻辑，复用你之前的 mapping
    param_mapping = {
        "task_type": "TaskType", "status": "Status", "is_time_out": "IsTimeOut",
        "is_active": "IsActive", "source_station_name": "SourceStationName",
        "dest_station_name": "DestStationName", "page": "page", "size": "size"
    }
    
    api_params = {}
    for k, v in params.items():
        if v is not None:
            api_params[param_mapping.get(k, k)] = v

    url = "http://106.15.57.43:8848/api/AI/Task"
    try:
        resp = requests.get(url, params=api_params, timeout=5)
        data = resp.json()
    except Exception as e:
        data = {"error": str(e)}
        
    return {"final_result": data}


# ==========================================
# 4. 构建图 (Topology) - 核心改动
# ==========================================

def route_intent(state: AgentState):
    """条件路由逻辑"""
    if state["intent"] == "NEW_QUERY":
        return "cleaner"
    else:
        return "extractor"

def build_graph():
    workflow = StateGraph(AgentState)

    # 注册节点
    workflow.add_node("classifier", intent_classifier_node)
    workflow.add_node("cleaner", state_cleaner_node) # 新增清洗节点
    workflow.add_node("extractor", extract_params_node)
    workflow.add_node("fetcher", fetch_data_node)

    # 定义流程图
    # 1. 入口 -> 分类器
    workflow.set_entry_point("classifier")
    
    # 2. 分类器 -> 条件分支 (是否去清洗)
    workflow.add_conditional_edges(
        "classifier",
        route_intent,
        {
            "cleaner": "cleaner",   # 走清洗路线
            "extractor": "extractor" # 走保留路线
        }
    )
    
    # 3. 清洗器 -> 提取器 (洗完后，当作白纸继续提取)
    workflow.add_edge("cleaner", "extractor")
    
    # 4. 提取器 -> 请求器
    workflow.add_edge("extractor", "fetcher")
    
    # 5. 请求器 -> 结束
    workflow.add_edge("fetcher", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


if __name__ == "__main__":
    app = build_graph()
    config = {"configurable": {"thread_id": "router_demo_01"}}
    
    print("="*40)
    print("🚀 智能调度助手 (Router + Cleaner 架构)")
    print("="*40)

    # 查询神经内科到静配中心的已到达的临时任务
    # 查询下这些任务当中的超时任务

    while True:
        q = input("\n👉 指令: ").strip()
        if q in ["q", "exit"]: break
        if not q: continue
        
        inputs = {"messages": [HumanMessage(content=q)]}
        res = app.invoke(inputs, config=config)
        
        print(f"🤖 结果: {json.dumps(res['final_result'], ensure_ascii=False, indent=2)}")