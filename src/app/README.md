# Shopping Assistant - Multi-Agent Architecture

Hệ thống trợ lý mua sắm sử dụng kiến trúc multi-agent với LangGraph.

## Kiến trúc

```
User Question
     ↓
Supervisor Agent (routing)
     ↓
     ├→ Policy Worker (RAG)
     ├→ Data Worker (lookup)
     └→ Response Worker (synthesis)
          ↓
     Final Answer
```

## Agents

### 1. Supervisor Agent
- **Nhiệm vụ**: Phân tích câu hỏi và định tuyến đến worker phù hợp
- **Output**: `needs_policy`, `needs_data`, `status`, `clarification_question`
- **File**: `graph.py` - `supervisor_node()`

### 2. Policy Worker (RAG Agent)
- **Nhiệm vụ**: Tìm kiếm thông tin chính sách từ vector store
- **Tools**: `search_policy` (RAG với Chroma)
- **Chunking**: Theo heading 2 + heading 3 + content
- **Embedding**: `sentence-transformers/all-MiniLM-L6-v2`
- **File**: `graph.py` - `policy_worker_node()`

### 3. Data Worker (Lookup Agent)
- **Nhiệm vụ**: Tra cứu dữ liệu đơn hàng, khách hàng, voucher
- **Tools**: 
  - `get_order_by_id`
  - `get_customer_by_id`
  - `get_voucher_by_code`
- **Data source**: `data/order_customer_mock_data.json`
- **File**: `graph.py` - `data_worker_node()`

### 4. Response Worker
- **Nhiệm vụ**: Tổng hợp kết quả và tạo câu trả lời cuối cùng
- **Input**: Policy info + Data info từ workers
- **Output**: Natural language response
- **File**: `graph.py` - `response_worker_node()`

## Files

- `graph.py` - LangGraph workflow và agent nodes
- `state.py` - Shared state schema
- `prompts.py` - System prompts cho từng agent
- `data_access.py` - Tools tra cứu data
- `config.py` - Settings và environment config
- `utils.py` - Helper functions
- `cli.py` - Command-line interface
- `streamlit_app.py` - Web UI

## State Schema

```python
class ShoppingState(TypedDict):
    messages: list
    question: str
    policy_info: str
    data_info: str
    final_answer: str
    status: str
    needs_policy: bool
    needs_data: bool
```

## Tools

### RAG Tool
- `search_policy(query: str) -> str`: Semantic search trên Chroma vector store

### Data Tools
- `get_order_by_id(order_id: int) -> dict`: Lấy thông tin đơn hàng
- `get_customer_by_id(customer_id: int) -> dict`: Lấy thông tin khách hàng
- `get_voucher_by_code(code: str) -> dict`: Lấy thông tin voucher

## Usage

### CLI Single Question
```bash
PYTHONPATH=src python -m app.cli --question "Đơn hàng 1971 có được hoàn trả không?"
```

### CLI Batch Test
```bash
PYTHONPATH=src python -m app.cli --batch --test-file data/test.json
```

### Streamlit UI
```bash
PYTHONPATH=src streamlit run src/app/streamlit_app.py
```

## Configuration

File `.env`:
```bash
LLM_MODEL=gemini-3.1-flash-lite
GOOGLE_API_KEY=your_key_here
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## Flow Logic

1. **Supervisor** nhận câu hỏi → phân tích → quyết định cần policy/data
2. **Policy Worker** (nếu cần) → RAG search → lưu vào `state.policy_info`
3. **Data Worker** (nếu cần) → lookup tools → lưu vào `state.data_info`
4. **Response Worker** → tổng hợp info → sinh `final_answer`
5. Return final answer

## Error Handling

- `clarification_needed`: Câu hỏi không rõ ràng
- `not_found`: Không tìm thấy thông tin
- Tool errors: Trả về error message trong tool response
