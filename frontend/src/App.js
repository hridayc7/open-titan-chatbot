import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import axios from 'axios';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [useTranslation, setUseTranslation] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Focus the input field when the component mounts
    inputRef.current?.focus();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    // Add user message
    const userMessage = { text: input, sender: 'user' };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      // Call API - make sure port matches your Flask backend
      // In App.js
      const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5001';

      // Then in your handleSubmit function
      const response = await axios.post(`${apiUrl}/api/chat`, {
        query: input,
        use_translation: useTranslation
      });

      // Extract data from response
      const { answer, sub_queries } = response.data;

      // Add sub-queries if present
      if (sub_queries && sub_queries.length > 0) {
        setMessages(prev => [
          ...prev,
          { text: 'I broke down your question into:', sender: 'bot', type: 'info' },
          { text: sub_queries.join('\n'), sender: 'bot', type: 'sub-queries' }
        ]);
      }

      // Add main answer
      setMessages(prev => [...prev, { text: answer, sender: 'bot' }]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        text: 'Sorry, I encountered an error. Please try again.',
        sender: 'bot',
        type: 'error'
      }]);
    } finally {
      setLoading(false);
      // Focus back on input after response
      inputRef.current?.focus();
    }
  };

  return (
    <div className="App">
      <aside className="App-sidebar">
        <div className="sidebar-header">
          <h2>OpenTitan</h2>
        </div>
        <div className="new-chat-button">
          <button onClick={() => setMessages([])}>
            <span className="plus-icon">+</span> New Chat
          </button>
        </div>
        <div className="translation-toggle">
          <label className="toggle-label">
            <span>Query Translation</span>
            <div className="toggle-switch-container">
              <input
                type="checkbox"
                checked={useTranslation}
                onChange={() => setUseTranslation(!useTranslation)}
              />
              <span className="toggle-switch"></span>
            </div>
          </label>
          <div className="toggle-description">
            Break complex queries into simpler questions
          </div>
        </div>
      </aside>

      <main className="App-main">
        <div className="chat-container">
          {messages.length === 0 ? (
            <div className="welcome-container">
              <h1>OpenTitan RAG Chatbot</h1>
              <div className="welcome-message">
                <p>Ask me anything about OpenTitan!</p>
              </div>
            </div>
          ) : (
            <div className="message-list">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`message-container ${message.sender}-container ${message.type || ''}`}
                >
                  <div className="message-avatar">
                    {message.sender === 'user' ? (
                      <div className="user-avatar">You</div>
                    ) : (
                      <div className="bot-avatar">AI</div>
                    )}
                  </div>
                  <div className="message-content">
                    <div className="message-text">
                      {message.text}
                    </div>
                  </div>
                </div>
              ))}

              {loading && (
                <div className="message-container bot-container">
                  <div className="message-avatar">
                    <div className="bot-avatar">AI</div>
                  </div>
                  <div className="message-content">
                    <div className="loading-indicator">
                      <span></span><span></span><span></span>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <div className="input-area">
          <form onSubmit={handleSubmit}>
            <div className="input-container">
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Message OpenTitan..."
                disabled={loading}
              />
              <button
                type="submit"
                className={loading || !input.trim() ? 'disabled' : ''}
                disabled={loading || !input.trim()}
              >
                <svg viewBox="0 0 24 24" fill="currentColor">
                  <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"></path>
                </svg>
              </button>
            </div>
            <div className="input-footer">
              OpenTitan RAG uses Claude to answer your questions based on retrieved documents
            </div>
          </form>
        </div>
      </main>
    </div>
  );
}

export default App;