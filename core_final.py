import requests
import json
from typing import Optional, Type
from pydantic import BaseModel, Field, field_validator
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

class TaskSearchInput(BaseModel):
    # --- 枚举/ID/状态 ---
    task_type: Optional[int] = Field(
        default=None, 
        description="精确匹配任务类型：1=临时任务, 2=固定任务, 3=空箱任务。例如用户说'查临时任务'时填1。"
    )
    status: Optional[int] = Field(
        default=None, 
        description="精确匹配任务状态：1=已创建, 2=运送中, 3=已到达, 4=已取消。例如用户说'查正在运送的任务'时填2。"
    )
    task_id: Optional[int] = Field(
        default=None, 
        description="精确任务ID。对应API参数 'Id'。"
    )
    is_time_out: Optional[bool] = Field(
        default=None, 
        description="是否超时 (True/False)。例如用户说'查超时的任务'时填 True。"
    )
    is_active: Optional[bool] = Field(
        default=None, 
        description="是否激活 (True/False)。"
    )

    # --- 名称/条码/摘要 ---
    box_bar_code: Optional[str] = Field(
        default=None, 
        description="箱子条码，支持模糊包含匹配。例如用户说'箱码包含A102'。"
    )
    source_station_name: Optional[str] = Field(
        default=None, 
        description="源站点名称，支持模糊匹配。例如'从 A 区发出的'。"
    )
    dest_station_name: Optional[str] = Field(
        default=None, 
        description="目标站点名称，支持模糊匹配。例如'送到 B 区的'。"
    )
    summary: Optional[str] = Field(
        default=None, 
        description="任务摘要/备注，支持模糊匹配。"
    )

    source_station_id: Optional[int] = Field(
        default=None, 
        description="源站点ID，支持模糊匹配。"
    )
    dest_station_id: Optional[int] = Field(
        default=None, 
        description="目标站点ID，支持模糊匹配。"
    )

    # --- 时间范围查询 (日期/时间) ---
    create_date_from: Optional[str] = Field(
        default=None, 
        description="创建时间范围-起始。格式如 '2025-05-01' 或 '2025-05-01 00:00:00'。"
    )
    create_date_to: Optional[str] = Field(
        default=None, 
        description="创建时间范围-结束。"
    )
    task_start_time_from: Optional[str] = Field(
        default=None, 
        description="任务实际开始时间的起始范围。"
    )
    task_start_time_to: Optional[str] = Field(
        default=None, 
        description="任务实际开始时间的结束范围。"
    )
    task_finish_time_from: Optional[str] = Field(
        default=None, 
        description="任务完成时间的起始范围。"
    )
    task_finish_time_to: Optional[str] = Field(
        default=None, 
        description="任务完成时间的结束范围。"
    )

    # --- 数值范围查询 (耗时/预估) ---
    min_consume_time: Optional[int] = Field(
        default=None, 
        description="最小消耗时间(秒)。例如用户说'耗时大于5分钟'，请计算为300并填入。"
    )
    max_consume_time: Optional[int] = Field(
        default=None, 
        description="最大消耗时间(秒)。"
    )
    min_estimated_time: Optional[int] = Field(
        default=None, 
        description="最小预估时间(秒)。"
    )
    max_estimated_time: Optional[int] = Field(
        default=None, 
        description="最大预估时间(秒)。"
    )

    # --- 分页参数 ---
    page: int = Field(default=1, description="页码，默认为1。")
    size: int = Field(default=10, description="每页大小，默认为10。")

class StationSearchInput(BaseModel):
    pass

class TaskQueryTool(BaseTool):
    name: str = "query_agv_tasks"
    description: str = "查询任务列表。支持按时间、状态、类型、站点、条码等多维度筛选。"
    args_schema: Type[BaseModel] = TaskSearchInput

    def _run(self, **kwargs) -> str:
        # 定义参数映射表 (Python字段名 -> API参数名)
        param_mapping = {
            "task_type": "TaskType",
            "status": "Status",
            "task_id": "Id",
            "is_time_out": "IsTimeOut",
            "is_active": "IsActive",
            
            # 模糊匹配
            "box_bar_code": "BoxBarCode",
            "source_station_name": "SourceStationName",
            "dest_station_name": "DestStationName",
            "summary": "Summary",
            "source_station_id": "SourceStationId",
            "dest_station_id": "DestStationId",
            
            # 时间范围
            "create_date_from": "CreateDateFrom",
            "create_date_to": "CreateDateTo",
            "task_start_time_from": "TaskStartTimeFrom",
            "task_start_time_to": "TaskStartTimeTo",
            "task_finish_time_from": "TaskFinishTimeFrom",
            "task_finish_time_to": "TaskFinishTimeTo",
            
            # 数值范围
            "min_consume_time": "MinConsumeTime",
            "max_consume_time": "MaxConsumeTime",
            "min_estimated_time": "MinEstimatedTime",
            "max_estimated_time": "MaxEstimatedTime",
            
            # 分页
            "page": "page",
            "size": "size"
        }

        api_params = {}
        print("\n[Tool] 大模型提取参数:")
        for py_key, value in kwargs.items():
            if value is not None:
                api_key = param_mapping.get(py_key, py_key)
                api_params[api_key] = value
                print(f"  - {py_key} ({value}) -> API: {api_key}")

        url = "http://106.15.57.43:8848/api/AI/Task"
        try:
            resp = requests.get(url, params=api_params, timeout=10)
            return json.dumps(resp.json(), ensure_ascii=False)
        except Exception as e:
            return f"API请求失败: {e}"

class StationQueryTool(BaseTool):
    name: str = "get_all_stations"
    
    description: str = "获取系统中所有的站点列表、位置坐标等信息。直接调用即可，不需要传入任何参数。"
    
    args_schema: Type[BaseModel] = StationSearchInput

    def _run(self) -> str:
        print(f"\n[Tool] 正在查询所有站点信息 (无参请求)...")
        
        url = "http://106.15.57.43:8848/api/AI/Task/Stations"
        resp = requests.get(url)
        return json.dumps(resp.json())

def run_advanced_agent():
    llm = ChatOpenAI(
        model="/model/qwen3-235b-a22b",
        openai_api_base="https://aimpapi.midea.com/t-aigc/aimp-qwen3-235b-a22b/v1",
        openai_api_key="msk-9e80428b8a8e4baa47e44ccb8dc96c4e1e59a80a0f2001b0d6efa63ed7b8ea76",
        temperature=0.01,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}}
    )

    tools = [TaskQueryTool(), StationQueryTool()]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个智能调度助手。请仔细分析用户的意图，选择合适的工具。"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    # query1 = "帮我查一下运送中的任务"
    # print(f"\n=== 用户提问 1: {query1} ===")
    # agent_executor.invoke({"input": query1})

    # query2 = "帮我查一下5月创建的空箱任务，消耗时间少于5分钟，箱码包含102"
    # print(f"\n=== 用户提问 2: {query2} ===")
    # agent_executor.invoke({"input": query2})
    
    query3 = "查询神经内科到静配中心的已到达临时任务"
    print(f"\n=== 用户提问 3: {query3} ===")
    result = agent_executor.invoke({"input": query3})
    print(f"\n=== 系统回复 3: {result} ===")

    # query4 = "现在都有哪些站点呢？"
    # print(f"\n=== 用户提问 4: {query4} ===")
    # agent_executor.invoke({"input": query4})

if __name__ == "__main__":
    run_advanced_agent()