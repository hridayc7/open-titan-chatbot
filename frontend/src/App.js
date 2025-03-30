import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';

function App() {
  // Main state
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [useTranslation, setUseTranslation] = useState(false);

  // UI state
  const [darkMode, setDarkMode] = useState(
    localStorage.getItem('darkMode') === 'true' || window.matchMedia('(prefers-color-scheme: dark)').matches
  );
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [currentChatId, setCurrentChatId] = useState('default');
  const [chatHistory, setChatHistory] = useState(() => {
    const savedHistory = localStorage.getItem('chatHistory');
    if (savedHistory) {
      try {
        return JSON.parse(savedHistory);
      } catch (e) {
        console.error('Failed to parse chat history:', e);
        return { default: { title: 'New Chat', messages: [], timestamp: Date.now() } };
      }
    } else {
      return { default: { title: 'New Chat', messages: [], timestamp: Date.now() } };
    }
  });
  const [chatMenuOpen, setChatMenuOpen] = useState(null);
  const [menuPosition, setMenuPosition] = useState({ top: 0, right: 0 });
  const [renameInput, setRenameInput] = useState('');
  const [isRenaming, setIsRenaming] = useState(false);

  // Refs
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const settingsRef = useRef(null);
  const chatMenuRef = useRef(null);
  const renameInputRef = useRef(null);

  // Auto-scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Focus input on load
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Apply dark mode
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark-mode');
    } else {
      document.documentElement.classList.remove('dark-mode');
    }
    localStorage.setItem('darkMode', darkMode);
  }, [darkMode]);

  // Load current chat messages
  useEffect(() => {
    if (chatHistory[currentChatId]) {
      setMessages(chatHistory[currentChatId].messages || []);
    } else {
      setMessages([]);
    }
  }, [currentChatId, chatHistory]);

  // Save chat history to localStorage
  useEffect(() => {
    localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
  }, [chatHistory]);

  // Close menus when clicking outside
  useEffect(() => {
    function handleClickOutside(event) {
      if (settingsRef.current && !settingsRef.current.contains(event.target)) {
        setSettingsOpen(false);
      }
      if (chatMenuRef.current && !chatMenuRef.current.contains(event.target)) {
        setChatMenuOpen(null);
        setIsRenaming(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  // Focus rename input when opening rename mode
  useEffect(() => {
    if (isRenaming) {
      renameInputRef.current?.focus();
    }
  }, [isRenaming]);

  // Toggle dark mode
  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  // Create a new chat
  const createNewChat = () => {
    const newChatId = 'chat_' + Date.now();
    setChatHistory(prev => ({
      [newChatId]: { title: 'New Chat', messages: [], timestamp: Date.now() },
      ...prev,
    }));
    setCurrentChatId(newChatId);
    setMessages([]);
    // Close any open menus
    setChatMenuOpen(null);
    setSettingsOpen(false);
  };

  // Delete a chat
  const deleteChat = (chatId) => {
    if (Object.keys(chatHistory).length <= 1) {
      // If this is the last chat, create a new empty one instead of deleting
      createNewChat();
      setChatHistory(prev => {
        const newHistory = { ...prev };
        delete newHistory[chatId];
        return newHistory;
      });
      return;
    }

    setChatHistory(prev => {
      const newHistory = { ...prev };
      delete newHistory[chatId];
      return newHistory;
    });

    // If we're deleting the current chat, switch to the newest one
    if (chatId === currentChatId) {
      // Find chat with newest timestamp
      const sortedChats = Object.entries(chatHistory)
        .filter(([id]) => id !== chatId)
        .sort((a, b) => b[1].timestamp - a[1].timestamp);

      if (sortedChats.length > 0) {
        setCurrentChatId(sortedChats[0][0]);
      }
    }

    // Close the menu
    setChatMenuOpen(null);
  };

  // Start renaming a chat
  const startRenameChat = (chatId) => {
    setRenameInput(chatHistory[chatId].title);
    setIsRenaming(true);
    setChatMenuOpen(chatId);
  };

  // Finish renaming a chat
  const finishRenameChat = (chatId) => {
    if (renameInput.trim()) {
      setChatHistory(prev => ({
        ...prev,
        [chatId]: {
          ...prev[chatId],
          title: renameInput.trim()
        }
      }));
    }
    setIsRenaming(false);
    setChatMenuOpen(null);
  };

  // Toggle chat menu
  const toggleChatMenu = (chatId, e) => {
    e.stopPropagation();

    // If opening menu or changing to different chat menu
    if (chatMenuOpen !== chatId) {
      // Calculate position from the clicked element
      const rect = e.currentTarget.getBoundingClientRect();
      setMenuPosition({
        top: rect.bottom + window.scrollY,
        right: window.innerWidth - rect.right,
      });
      setChatMenuOpen(chatId);
    } else {
      // Closing the menu
      setChatMenuOpen(null);
    }
  };

  // Rename current chat based on content
  const updateChatTitle = (messages) => {
    if (messages.length > 0 && messages[0].sender === 'user') {
      // Use first user message as title, truncate if needed
      let title = messages[0].text;
      if (title.length > 30) {
        title = title.substring(0, 27) + '...';
      }

      setChatHistory(prev => ({
        ...prev,
        [currentChatId]: {
          ...prev[currentChatId],
          title,
          messages
        }
      }));
    }
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    // Add user message
    const userMessage = { text: input, sender: 'user' };
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);

    // Update chat history
    setChatHistory(prev => ({
      ...prev,
      [currentChatId]: {
        ...prev[currentChatId],
        messages: updatedMessages,
        timestamp: Date.now()
      }
    }));

    // Update the chat title based on first message
    if (messages.length === 0) {
      updateChatTitle(updatedMessages);
    }

    setInput('');
    setLoading(true);

    try {
      // Call API
      const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5001';
      const response = await axios.post(`${apiUrl}/api/chat`, {
        query: input,
        use_translation: useTranslation
      });

      // Extract data from response
      const { answer, sub_queries } = response.data;
      let finalMessages = [...updatedMessages];

      // Add sub-queries if present
      if (sub_queries && sub_queries.length > 0) {
        const infoMessage = { text: 'I broke down your question into:', sender: 'bot', type: 'info' };
        const queriesMessage = { text: sub_queries.join('\n'), sender: 'bot', type: 'sub-queries' };

        finalMessages = [...finalMessages, infoMessage, queriesMessage];

        setMessages(finalMessages);
      }

      // Add main answer
      const botMessage = { text: answer, sender: 'bot' };
      finalMessages = [...finalMessages, botMessage];

      setMessages(finalMessages);

      // Update chat history with all messages
      setChatHistory(prev => ({
        ...prev,
        [currentChatId]: {
          ...prev[currentChatId],
          messages: finalMessages,
          timestamp: Date.now()
        }
      }));
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = {
        text: 'Sorry, I encountered an error. Please try again.',
        sender: 'bot',
        type: 'error'
      };

      const finalMessages = [...updatedMessages, errorMessage];
      setMessages(finalMessages);

      setChatHistory(prev => ({
        ...prev,
        [currentChatId]: {
          ...prev[currentChatId],
          messages: finalMessages,
          timestamp: Date.now()
        }
      }));
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  // Fix duplicate source URL display issue
  const fixSourceUrlText = (text) => {
    return text.replace(/\[([^\]]+)\]\(([^)]+)\)\]\(([^)]+)\)/g, '[$1]($2)');
  };

  // Sort chats by timestamp (newest first)
  const sortedChats = Object.entries(chatHistory)
    .sort((a, b) => b[1].timestamp - a[1].timestamp);

  return (
    <div className={`App ${darkMode ? 'dark-theme' : 'light-theme'}`}>
      <aside className="App-sidebar">
        <div className="sidebar-header">
          <h2>OpenTitan</h2>
        </div>

        <div className="new-chat-button">
          <button onClick={createNewChat}>
            <span className="plus-icon">+</span> New Chat
          </button>
        </div>

        <div className="chat-history">
          {sortedChats.map(([chatId, chat]) => (
            <div
              key={chatId}
              className={`chat-history-item ${currentChatId === chatId ? 'active' : ''}`}
              onClick={() => setCurrentChatId(chatId)}
            >
              <span className="chat-title">{chat.title}</span>
              <button
                className="chat-menu-btn"
                onClick={(e) => toggleChatMenu(chatId, e)}
              >
                <svg viewBox="0 0 24 24" width="18" height="18">
                  <circle cx="5" cy="12" r="2" fill="currentColor" />
                  <circle cx="12" cy="12" r="2" fill="currentColor" />
                  <circle cx="19" cy="12" r="2" fill="currentColor" />
                </svg>
              </button>
            </div>
          ))}
        </div>
      </aside>

      {/* Render menu at the app root level for proper z-index stacking */}
      {chatMenuOpen && (
        <div
          className="chat-menu-container"
          ref={chatMenuRef}
          style={{
            position: 'fixed',
            top: `${menuPosition.top}px`,
            right: `${menuPosition.right}px`,
          }}
        >
          <div className="chat-menu">
            {isRenaming ? (
              <div className="rename-container">
                <input
                  ref={renameInputRef}
                  type="text"
                  value={renameInput}
                  onChange={(e) => setRenameInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      finishRenameChat(chatMenuOpen);
                    } else if (e.key === 'Escape') {
                      setIsRenaming(false);
                      setChatMenuOpen(null);
                    }
                  }}
                  className="rename-input"
                />
                <button
                  className="rename-btn"
                  onClick={() => finishRenameChat(chatMenuOpen)}
                >
                  Save
                </button>
              </div>
            ) : (
              <>
                <button
                  className="menu-item"
                  onClick={() => startRenameChat(chatMenuOpen)}
                >
                  <svg viewBox="0 0 24 24" width="16" height="16">
                    <path fill="currentColor" d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04c.39-.39.39-1.02 0-1.41l-2.34-2.34c-.39-.39-1.02-.39-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z" />
                  </svg>
                  Rename
                </button>
                <button
                  className="menu-item delete-item"
                  onClick={() => deleteChat(chatMenuOpen)}
                >
                  <svg viewBox="0 0 24 24" width="16" height="16">
                    <path fill="currentColor" d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z" />
                  </svg>
                  Delete
                </button>
              </>
            )}
          </div>
        </div>
      )}

      <main className="App-main">
        <div className="main-header">
          <h3 className="current-chat-title">
            {chatHistory[currentChatId]?.title || 'New Chat'}
          </h3>
          <div className="settings-container" ref={settingsRef}>
            <button
              className="settings-button"
              onClick={() => setSettingsOpen(!settingsOpen)}
            >
              <svg viewBox="0 0 24 24" width="24" height="24">
                <path
                  fill="currentColor"
                  d="M12,15.5A3.5,3.5 0 0,1 8.5,12A3.5,3.5 0 0,1 12,8.5A3.5,3.5 0 0,1 15.5,12A3.5,3.5 0 0,1 12,15.5M19.43,12.97C19.47,12.65 19.5,12.33 19.5,12C19.5,11.67 19.47,11.34 19.43,11L21.54,9.37C21.73,9.22 21.78,8.95 21.66,8.73L19.66,5.27C19.54,5.05 19.27,4.96 19.05,5.05L16.56,6.05C16.04,5.66 15.5,5.32 14.87,5.07L14.5,2.42C14.46,2.18 14.25,2 14,2H10C9.75,2 9.54,2.18 9.5,2.42L9.13,5.07C8.5,5.32 7.96,5.66 7.44,6.05L4.95,5.05C4.73,4.96 4.46,5.05 4.34,5.27L2.34,8.73C2.21,8.95 2.27,9.22 2.46,9.37L4.57,11C4.53,11.34 4.5,11.67 4.5,12C4.5,12.33 4.53,12.65 4.57,12.97L2.46,14.63C2.27,14.78 2.21,15.05 2.34,15.27L4.34,18.73C4.46,18.95 4.73,19.03 4.95,18.95L7.44,17.94C7.96,18.34 8.5,18.68 9.13,18.93L9.5,21.58C9.54,21.82 9.75,22 10,22H14C14.25,22 14.46,21.82 14.5,21.58L14.87,18.93C15.5,18.67 16.04,18.34 16.56,17.94L19.05,18.95C19.27,19.03 19.54,18.95 19.66,18.73L21.66,15.27C21.78,15.05 21.73,14.78 21.54,14.63L19.43,12.97Z"
                />
              </svg>
            </button>
            {settingsOpen && (
              <div className="settings-dropdown">
                <div className="settings-item">
                  <span>Dark Mode</span>
                  <label className="toggle-switch-container">
                    <input
                      type="checkbox"
                      checked={darkMode}
                      onChange={toggleDarkMode}
                    />
                    <span className="toggle-switch"></span>
                  </label>
                </div>
              </div>
            )}
          </div>
        </div>

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
                      {message.sender === 'bot' && !message.type ? (
                        <ReactMarkdown
                          components={{
                            a: ({ node, ...props }) => (
                              <a {...props} target="_blank" rel="noopener noreferrer" className="source-link" />
                            ),
                            code: ({ node, inline, className, children, ...props }) => {
                              const match = /language-(\w+)/.exec(className || '');
                              return !inline && match ? (
                                <div className="code-block-container">
                                  <div className="code-block-header">
                                    <span className="code-language">{match[1]}</span>
                                    <button
                                      className="copy-button"
                                      onClick={() => {
                                        navigator.clipboard.writeText(String(children).replace(/\n$/, ''));
                                      }}
                                    >
                                      Copy
                                    </button>
                                  </div>
                                  <SyntaxHighlighter
                                    language={match[1]}
                                    style={darkMode ? tomorrow : oneLight}
                                    PreTag="div"
                                    {...props}
                                  >
                                    {String(children).replace(/\n$/, '')}
                                  </SyntaxHighlighter>
                                </div>
                              ) : (
                                <code className={className} {...props}>
                                  {children}
                                </code>
                              );
                            }
                          }}
                        >
                          {fixSourceUrlText(message.text)}
                        </ReactMarkdown>
                      ) : (
                        message.text
                      )}
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
              <div className="input-options">
                <label className="multi-query-toggle">
                  <input
                    type="checkbox"
                    checked={useTranslation}
                    onChange={() => setUseTranslation(!useTranslation)}
                  />
                  <span className="toggle-text">Advanced Reasoning</span>
                </label>
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