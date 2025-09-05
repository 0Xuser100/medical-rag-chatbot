from flask import Flask,render_template,request,session,redirect,url_for,jsonify
from components.memory import create_session_qa_chain
from config.config import OPEN_AI_API_KEY,public_key,secret_key,host
import os
import traceback
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
langfuse = Langfuse(
    public_key=public_key,
    secret_key=secret_key,
    host=host
)
langfuse_handler = CallbackHandler()

app = Flask(__name__)
app.secret_key = os.urandom(24)

def get_qa_chain_with_memory(session_messages=None):
    """Get QA chain with conversation memory"""
    try:
        # Create QA chain with memory populated from session
        return create_session_qa_chain(session_messages)
    except Exception as e:
        print(f"Failed to create QA chain with memory: {e}")
        raise

from markupsafe import Markup
def nl2br(value):
    return Markup(value.replace("\n" , "<br>\n"))

app.jinja_env.filters['nl2br'] = nl2br

@app.route("/" , methods=["GET","POST"])
def index():
    if "messages" not in session:
        session["messages"]=[]

    if request.method=="POST":
        user_input = request.form.get("prompt")

        if user_input:
            messages = session["messages"]
            messages.append({"role" : "user" , "content":user_input})
            session["messages"] = messages

            try:
                # Check if API key is configured
                if not OPEN_AI_API_KEY:
                    error_msg = "OpenAI API key not configured. Please check your .env file."
                    return render_template("index.html", messages=session["messages"], error=error_msg)
                
                # Get QA chain with conversation memory
                qa_chain = get_qa_chain_with_memory(session.get("messages", []))
                
                # Invoke the QA chain with question
                response = qa_chain.invoke({"question": user_input},config={"callbacks": [langfuse_handler]})
                result = response.get("answer", "Sorry, I couldn't generate a response.")
                
                # Add assistant response to messages
                messages.append({"role": "assistant", "content": result})
                session["messages"] = messages
                
            except Exception as e:
                # Detailed error logging
                error_details = traceback.format_exc()
                print(f"Error in chatbot: {error_details}")
                
                # User-friendly error message
                if "API" in str(e) or "OpenAI" in str(e):
                    error_msg = "API connection error. Please check your OpenAI API key and try again."
                elif "vector" in str(e).lower() or "faiss" in str(e).lower():
                    error_msg = "Database error. Please contact administrator."
                else:
                    error_msg = f"An error occurred: {str(e)}"
                    
                return render_template("index.html", messages=session["messages"], error=error_msg)
            
        return redirect(url_for("index"))
    return render_template("index.html" , messages=session.get("messages" , []))

@app.route("/clear")
def clear():
    """Clear conversation history and reset session"""
    session.pop("messages", None)
    session.pop("conversation_id", None)  # Clear conversation tracking
    return redirect(url_for("index"))

@app.route("/export")
def export_conversation():
    """Export conversation history as text"""
    messages = session.get("messages", [])
    if not messages:
        return "No conversation to export", 400
    
    # Format conversation for export
    export_text = "Medical AI Assistant Conversation\n"
    export_text += "=" * 50 + "\n\n"
    
    for msg in messages:
        role = "You" if msg["role"] == "user" else "AI Assistant"
        export_text += f"{role}: {msg['content']}\n\n"
    
    from flask import Response
    return Response(
        export_text,
        mimetype="text/plain",
        headers={"Content-Disposition": "attachment;filename=medical_conversation.txt"}
    )

if __name__=="__main__":
    app.run(host="0.0.0.0" , port=5000 , debug=True , use_reloader = True)



