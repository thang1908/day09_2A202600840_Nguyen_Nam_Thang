import streamlit as st
from app.graph import ShoppingAssistant

st.set_page_config(page_title="Shopping Assistant", page_icon="🛒", layout="wide")

st.title("🛒 Shopping Assistant")
st.markdown("Hỏi về chính sách, đơn hàng, khách hàng hoặc voucher")

# Initialize assistant
if "assistant" not in st.session_state:
    st.session_state.assistant = ShoppingAssistant()

# Input
question = st.text_input("Câu hỏi của bạn:", placeholder="VD: Đơn hàng 1971 bao giờ được giao?")

if st.button("Gửi", type="primary") and question:
    with st.spinner("Đang xử lý..."):
        result = st.session_state.assistant.ask(question)
        
        # Display answer
        st.success("**Câu trả lời:**")
        answer = result.get("final_answer", "")
        if isinstance(answer, list):
            answer = answer[0].get("text", "") if answer else ""
        st.markdown(answer)
        
        # Display reasoning flow
        with st.expander("🔍 Luồng suy luận", expanded=True):
            trace = result.get("trace", [])
            
            for i, step in enumerate(trace, 1):
                node = step.get("node", "unknown")
                output = step.get("output", {})
                
                st.markdown(f"### Bước {i}: `{node}`")
                
                if node == "supervisor":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Status", output.get("status", ""))
                    with col2:
                        st.metric("Needs Policy", "✅" if output.get("needs_policy") else "❌")
                    with col3:
                        st.metric("Needs Data", "✅" if output.get("needs_data") else "❌")
                    
                    if output.get("clarification_question"):
                        st.warning(f"❓ {output['clarification_question']}")
                
                elif node in ["worker_1_policy", "worker_2_data"]:
                    messages = output.get("messages", [])
                    if messages:
                        for msg in messages:
                            if msg.get("type") == "tool":
                                st.code(msg.get("content", "")[:500] + "...", language="json")
                            elif msg.get("type") == "ai" and msg.get("content"):
                                content = msg["content"]
                                if isinstance(content, list):
                                    content = content[0].get("text", "") if content else ""
                                st.info(content[:300] + "...")
                
                elif node == "worker_3_response":
                    if isinstance(output, list):
                        output = output[0].get("text", "") if output else ""
                    st.success(output[:300] + "...")
                
                st.divider()
        
        # Display raw state
        with st.expander("📊 Raw State"):
            st.json(result, expanded=False)
