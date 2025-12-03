import requests
import json
import operator
from typing import Annotated, Dict, Any, Union, Optional
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from pydantic import BaseModel, Field, field_validator # 确保导入了 field_validator

# --- LangGraph 核心组件 ---
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ==========================================
# 1. 定义状态 (State) - 这是 LangGraph 的核心
# ==========================================

class AgentState(TypedDict):
    # 消息历史 (用于让模型理解上下文语义)
    messages: Annotated[list, operator.add] 
    
    # 【关键】当前的查询参数状态
    # 我们把参数存在 State 里，这样多轮对话就能自动继承上一轮的参数！
    current_params: Dict[str, Any] 
    
    # 最终的 API 结果 (JSON)
    final_result: Dict[str, Any]

# ==========================================
# 2. 定义 Pydantic 结构 (复用你之前的定义)
# ==========================================
# 为了节省 token，我简化了 description，实际使用请保留你详细的 description
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

    @field_validator('*', mode='before')
    @classmethod
    def parse_nested_dict(cls, v: Any) -> Any:
        # 检查是否是字典
        if isinstance(v, dict):
            # 情况 1: 模型返回了 {'value': 1} -> 拆包取值
            if 'value' in v:
                return v['value']
            
            # 情况 2: 模型返回了 {} (空字典) -> 视为 None (即未提取到有效值)
            # 这就是解决你当前报错的关键
            if not v:
                return None
            
            # 情况 3: 其他奇怪的字典 -> 为了不崩，统一转 None
            return None
            
        # 不是字典，直接返回（比如已经是正常的 int 或 str）
        return v

# ==========================================
# 3. 定义节点逻辑 (Nodes)
# ==========================================

# --- 节点 A: 参数提取与合并 ---
def extract_params_node(state: AgentState):
    """
    参数提取节点
    优化策略：只发送用户最新的一条指令给模型，避免历史对话干扰模型的判断。
    """
    llm = ChatOpenAI(
        model="/model/qwen3-235b-a22b",
        openai_api_base="https://aimpapi.midea.com/t-aigc/aimp-qwen3-235b-a22b/v1",
        openai_api_key="msk-9e80428b8a8e4baa47e44ccb8dc96c4e1e59a80a0f2001b0d6efa63ed7b8ea76",
        temperature=0.01,
        # 强制开启思考模式，有助于复杂逻辑推理
        extra_body={"chat_template_kwargs": {"enable_thinking": True}}
    )
    
    # 绑定 Pydantic 结构
    structured_llm = llm.with_structured_output(TaskSearchInput)
    
    # 获取当前参数
    current_p = state.get("current_params", {})
    
    # 获取用户最新的一条输入
    # state["messages"] 包含了所有的对话历史，我们只取最后一条 HumanMessage
    last_message = state["messages"][-1]
    
    # 构造更具有指向性的 System Prompt
    system_prompt = f"""
    你是一个状态更新器。你维护着一组查询参数。
    
    【当前生效参数】：
    {json.dumps(current_p, ensure_ascii=False)}
    
    【用户最新指令】：
    "{last_message.content}"
    
    请根据用户的指令，输出需要【修改或新增】的参数：
    1. 如果用户说“已超时的”，请输出 {{"is_time_out": true}}。
    2. 如果用户说“不要超时的”，请输出 {{"is_time_out": false}}。
    3. 如果用户说“去掉超时条件”，请输出 {{"is_time_out": null}}。
    4. 对于用户【未提及】的条件，请不要输出（即返回 null），系统会自动保留原值。
    """
    
    # 关键修改：只发 System Prompt，不发历史 messages
    # 这样模型就不会被之前的对话干扰，只专注处理当前这一句
    messages = [SystemMessage(content=system_prompt)]
    
    # 调用模型
    try:
        new_params_obj = structured_llm.invoke(messages)
        
        # 使用 model_dump(exclude_none=True) 过滤掉模型没填的字段
        new_params_dict = new_params_obj.model_dump(exclude_none=True)
        
        # --- 调试日志 (关键) ---
        print(f"\n[Debug] 用户指令: {last_message.content}")
        print(f"[Debug] 模型提取结果 (Raw): {new_params_dict}")
        # ---------------------

        # 合并参数：旧参数 update 新参数
        merged_params = current_p.copy()
        merged_params.update(new_params_dict)
        
        return {
            "current_params": merged_params
        }
        
    except Exception as e:
        print(f"\n[Error] 参数提取失败: {e}")
        # 如果出错，保持参数不变
        return {"current_params": current_p}

# --- 节点 B: API 调用 (Data Fetcher) ---
def fetch_data_node(state: AgentState):
    """
    这个节点的作用：纯粹的执行。
    拿到 state['current_params'] -> 调用 requests -> 存入 state['final_result']
    """
    params = state["current_params"]
    print(f"  >>> [API执行] 最终请求参数: {params}")
    
    # 1. 参数映射 (Snake -> Pascal)
    # 这里复用你之前的映射逻辑
    param_mapping = {
            "task_type": "TaskType", "status": "Status", "task_id": "Id",
            "is_time_out": "IsTimeOut", "is_active": "IsActive",
            "box_bar_code": "BoxBarCode", "source_station_name": "SourceStationName",
            "dest_station_name": "DestStationName", "summary": "Summary",
            "source_station_id": "SourceStationId", "dest_station_id": "DestStationId",
            "create_date_from": "CreateDateFrom", "create_date_to": "CreateDateTo",
            "task_start_time_from": "TaskStartTimeFrom", "task_start_time_to": "TaskStartTimeTo",
            "task_finish_time_from": "TaskFinishTimeFrom", "task_finish_time_to": "TaskFinishTimeTo",
            "min_consume_time": "MinConsumeTime", "max_consume_time": "MaxConsumeTime",
            "min_estimated_time": "MinEstimatedTime", "max_estimated_time": "MaxEstimatedTime",
            "page": "page", "size": "size"
        }
    
    api_params = {}
    for k, v in params.items():
        if v is not None:
            api_key = param_mapping.get(k, k)
            api_params[api_key] = v

    # 2. 发起请求
    url = "http://106.15.57.43:8848/api/AI/Task"

    try:
        resp = requests.get(url, params=api_params, timeout=10)
        data = resp.json()
    except Exception as e:
        data = {"error": str(e)}
        
    # 3. 更新 State
    return {"final_result": data}

# ==========================================
# 4. 构建图 (Graph Construction)
# ==========================================

def build_graph():
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("extract_params", extract_params_node)
    workflow.add_node("fetch_data", fetch_data_node)

    # 定义流程
    # Start -> extract_params -> fetch_data -> End
    workflow.set_entry_point("extract_params")
    workflow.add_edge("extract_params", "fetch_data")
    workflow.add_edge("fetch_data", END)

    # 设置记忆 (Checkpointer)
    # 这使得 graph.invoke 可以传入 thread_id 来恢复状态
    memory = MemorySaver()
    
    return workflow.compile(checkpointer=memory)

# ==========================================
# 5. 运行测试
# ==========================================

if __name__ == "__main__":
    app = build_graph()
    
    # 2. 配置线程 ID 
    # 固定这个 ID，就能在当前运行期间保持多轮对话记忆
    # 每次重新运行脚本，记忆会重置（因为用的是 MemorySaver 内存存储）
    thread_id = "console_user_001"
    config = {"configurable": {"thread_id": thread_id}}
    
    print("="*50)
    print("🏥 智能调度助手 - 命令行交互模式")
    print(f"当前会话 ID: {thread_id}")
    print("提示：输入 'q', 'quit', 'exit' 可退出程序")
    print("="*50)

    # 查询神经内科到静配中心的已到达的临时任务
    # 查询下这些任务当中的超时任务

    while True:
        try:
            # A. 获取用户输入
            user_query = input("\n👉 请输入指令: ").strip()
            
            # B. 检查退出条件
            if user_query.lower() in ["q", "quit", "exit"]:
                print("\n👋 程序已退出，再见！")
                break
            
            if not user_query:
                continue

            # C. 构造 Graph 输入
            # LangGraph 需要一个 messages 列表作为输入
            inputs = {"messages": [HumanMessage(content=user_query)]}
            
            # D. 执行调用
            # print("   (正在请求 API...)") # 可选：加个加载提示
            final_state = app.invoke(inputs, config=config)
            
            # E. 打印纯 JSON 结果
            result = final_state.get("final_result", {})
            
            print("\n🤖 [API JSON 响应]:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            
        except KeyboardInterrupt:
            # 捕获 Ctrl+C
            print("\n\n👋 用户强制中断")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")