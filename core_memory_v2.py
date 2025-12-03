import requests
import json
import operator
from typing import Annotated, Dict, Any, Union, Optional
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from pydantic import BaseModel, Field, field_validator # 确保导入了 field_validator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

class AgentState(TypedDict):
    messages: Annotated[list, operator.add] 

    current_params: Dict[str, Any] 
    
    final_result: Dict[str, Any]

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
        if isinstance(v, dict):
            if 'value' in v:
                return v['value']
            
            if not v:
                return None
            
            return None
            
        return v

def extract_params_node(state: AgentState):
    """
    参数提取节点
    优化策略：只发送用户最新的一条指令给模型，避免历史对话干扰模型的判断。
    """
    llm = ChatOpenAI(
        model="/model/qwen3-235b-a22b",
        openai_api_base="https://aimpapi.midea.com/t-aigc/aimp-deepseek-r1/v1",
        openai_api_key="msk-ca04571203246b31eec2dae635521ea079ca23818fdbe1f2177e17934382d378",
        temperature=0.01
        # extra_body={"chat_template_kwargs": {"enable_thinking": True}}
    )
    
    structured_llm = llm.with_structured_output(TaskSearchInput)
    
    current_p = state.get("current_params", {})
    
    last_message = state["messages"][-1]
    
    system_prompt = f"""
    你是一个状态更新器。你维护着一组查询参数。
    
    【当前生效参数】：
    {json.dumps(current_p, ensure_ascii=False)}
    
    【用户最新指令】：
    "{last_message.content}"
    
    请根据用户的指令，输出需要【修改或新增】的参数.
    """
    
    messages = [SystemMessage(content=system_prompt)]
    
    try:
        new_params_obj = structured_llm.invoke(messages)
        
        new_params_dict = new_params_obj.model_dump(exclude_none=True)
        
        print(f"\n[Debug] 用户指令: {last_message.content}")
        print(f"[Debug] 模型提取结果 (Raw): {new_params_dict}")

        merged_params = current_p.copy()
        merged_params.update(new_params_dict)
        
        return {
            "current_params": merged_params
        }
        
    except Exception as e:
        print(f"\n[Error] 参数提取失败: {e}")
        return {"current_params": current_p}

def fetch_data_node(state: AgentState):
    """
    这个节点的作用：纯粹的执行。
    拿到 state['current_params'] -> 调用 requests -> 存入 state['final_result']
    """
    params = state["current_params"]
    print(f"  >>> [API执行] 最终请求参数: {params}")
    
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

    url = "http://106.15.57.43:8848/api/AI/Task"

    try:
        resp = requests.get(url, params=api_params, timeout=10)
        data = resp.json()
    except Exception as e:
        data = {"error": str(e)}
        
    return {"final_result": data}

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("extract_params", extract_params_node)
    workflow.add_node("fetch_data", fetch_data_node)

    workflow.set_entry_point("extract_params")
    workflow.add_edge("extract_params", "fetch_data")
    workflow.add_edge("fetch_data", END)

    memory = MemorySaver()
    
    return workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    app = build_graph()
    
    thread_id = "console_user_001"
    config = {"configurable": {"thread_id": thread_id}}
    
    print("="*50)
    print("智能调度助手 - 命令行交互模式")
    print(f"当前会话 ID: {thread_id}")
    print("提示：输入 'q', 'quit', 'exit' 可退出程序")
    print("="*50)

    # 查询神经内科到静配中心的已到达的临时任务
    # 查询下这些任务当中的超时任务

    while True:
        try:
            user_query = input("\n请输入指令: ").strip()
            
            if user_query.lower() in ["q", "quit", "exit"]:
                print("\n程序已退出")
                break
            
            if not user_query:
                continue

            inputs = {"messages": [HumanMessage(content=user_query)]}
            
            final_state = app.invoke(inputs, config=config)
            
            result = final_state.get("final_result", {})
            
            print("\n[API JSON 响应]:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            
        except KeyboardInterrupt:
            print("\n\n用户强制中断")
            break
        except Exception as e:
            print(f"发生错误: {e}")