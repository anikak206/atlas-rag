'use client';

import { useState, useEffect, useRef } from 'react';
import { Send, Plus, FileText, BarChart3, Settings, Menu, X, Sparkles } from 'lucide-react';
import { api, setAuthToken } from '@/lib/api';
import { useAppStore } from '@/lib/store';
import { toast } from 'sonner';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: any[];
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { currentRole, setCurrentRole, currentTenant, setCurrentTenant } = useAppStore();
  const [isStreaming, setIsStreaming] = useState(false);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Auto-login for demo
    const login = async () => {
      try {
        const res = await api.post('/auth/login', {
          tenant_slug: currentTenant,
          email: currentRole === 'Admin' ? 'admin@acme.test' : currentRole === 'Finance' ? 'finance@acme.test' : 'intern@acme.test',
          password: 'ChangeMe-Acme-2026',
        });
        setAuthToken(res.data.tokens.access_token);
        localStorage.setItem('user', JSON.stringify(res.data.user));
        localStorage.setItem('tenant_id', res.data.tenant.id);
      } catch (e) {
        console.error('Login failed:', e);
      }
    };
    login();
  }, [currentRole, currentTenant]);

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);
    setIsStreaming(true);

    try {
      // Try streaming endpoint
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}${process.env.NEXT_PUBLIC_API_PREFIX}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
        },
        body: JSON.stringify({ question: input, include_sources: true }),
      });

      if (!response.ok) throw new Error('Query failed');

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let assistantMsg: Message = { id: (Date.now() + 1).toString(), role: 'assistant', content: '', citations: [] };

      setMessages((prev) => [...prev, assistantMsg]);

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                if (data.type === 'token') {
                  assistantMsg.content += data.content;
                  setMessages((prev) => {
                    const newMsgs = [...prev];
                    newMsgs[newMsgs.length - 1] = { ...assistantMsg };
                    return newMsgs;
                  });
                } else if (data.type === 'citations') {
                  assistantMsg.citations = data.citations;
                  setMessages((prev) => {
                    const newMsgs = [...prev];
                    newMsgs[newMsgs.length - 1] = { ...assistantMsg };
                    return newMsgs;
                  });
                }
              } catch (e) {
                // Non-JSON line
              }
            }
          }
        }
      } else {
        const data = await response.json();
        assistantMsg.content = data.answer;
        assistantMsg.citations = data.sources;
        setMessages((prev) => {
          const newMsgs = [...prev];
          newMsgs[newMsgs.length - 1] = assistantMsg;
          return newMsgs;
        });
      }
    } catch (e) {
      toast.error('Failed to get response');
      setMessages((prev) => prev.slice(0, -1));
    } finally {
      setIsLoading(false);
      setIsStreaming(false);
    }
  };

  return (
    <div className="flex h-screen bg-[#0a0a0a]">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'w-72' : 'w-0'} transition-all duration-300 bg-[#111111] border-r border-gray-800 flex flex-col overflow-hidden`}>
        <div className="p-4 border-b border-gray-800">
          <button className="w-full flex items-center gap-2 px-3 py-2 bg-[#1a1a1a] hover:bg-[#252525] rounded-lg transition-colors">
            <Plus size={18} />
            <span>New Chat</span>
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-2">
          <div className="text-xs text-gray-500 px-2 py-1">Recent Conversations</div>
        </div>
        <div className="p-4 border-t border-gray-800 space-y-2">
          <a href="/documents" className="flex items-center gap-2 px-3 py-2 hover:bg-[#1a1a1a] rounded-lg transition-colors">
            <FileText size={18} />
            <span>Knowledge Base</span>
          </a>
          <a href="/audit" className="flex items-center gap-2 px-3 py-2 hover:bg-[#1a1a1a] rounded-lg transition-colors">
            <BarChart3 size={18} />
            <span>Audit Dashboard</span>
          </a>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Top Bar */}
        <div className="h-16 border-b border-gray-800 flex items-center justify-between px-4">
          <div className="flex items-center gap-3">
            <button onClick={() => setSidebarOpen(!sidebarOpen)} className="p-2 hover:bg-[#1a1a1a] rounded-lg">
              {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
            <div className="flex items-center gap-2">
              <Sparkles className="text-blue-400" size={20} />
              <span className="font-semibold">Atlas RAG</span>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <select value={currentTenant} onChange={(e) => setCurrentTenant(e.target.value)} className="bg-[#1a1a1a] border border-gray-700 rounded-lg px-3 py-1.5 text-sm">
              <option value="acme">Acme Corp</option>
            </select>
            <select value={currentRole} onChange={(e) => setCurrentRole(e.target.value as any)} className="bg-[#1a1a1a] border border-gray-700 rounded-lg px-3 py-1.5 text-sm">
              <option value="Admin">Admin</option>
              <option value="Finance">Finance</option>
              <option value="Intern">Intern</option>
            </select>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <div className="h-full flex items-center justify-center">
              <div className="text-center space-y-4">
                <Sparkles className="w-12 h-12 mx-auto text-blue-400" />
                <h1 className="text-2xl font-semibold">Welcome to Atlas RAG</h1>
                <p className="text-gray-400">Ask anything about your knowledge base</p>
              </div>
            </div>
          ) : (
            <div className="max-w-4xl mx-auto py-8 px-4 space-y-6">
              {messages.map((msg) => (
                <div key={msg.id} className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-3xl space-y-3 ${msg.role === 'user' ? 'bg-blue-600 px-4 py-2 rounded-2xl' : ''}`}>
                    <div className="whitespace-pre-wrap">{msg.content || (isStreaming && '...' )}</div>
                    {msg.role === 'assistant' && msg.citations && msg.citations.length > 0 && (
                      <div className="space-y-2 pt-2 border-t border-gray-800">
                        <div className="text-xs text-gray-500">Sources</div>
                        <div className="flex flex-wrap gap-2">
                          {msg.citations.slice(0, 3).map((cite: any, idx: number) => (
                            <button key={idx} className="flex items-center gap-1 px-2 py-1 bg-[#1a1a1a] hover:bg-[#252525] rounded text-xs border border-gray-700">
                              <FileText size={12} />
                              <span>{cite.document_title?.slice(0, 30)}...</span>
                              {cite.page_number && <span className="text-gray-500">p.{cite.page_number}</span>}
                              {cite.score && <span className="text-green-400">{Math.round(cite.score * 100)}%</span>}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input */}
        <div className="border-t border-gray-800 p-4">
          <div className="max-w-4xl mx-auto flex gap-2">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && sendMessage()}
              placeholder="Ask a question..."
              className="flex-1 bg-[#1a1a1a] border border-gray-700 rounded-lg px-4 py-3 focus:outline-none focus:border-blue-500"
              disabled={isLoading}
            />
            <button onClick={sendMessage} disabled={isLoading || !input.trim()} className="px-4 py-3 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 rounded-lg transition-colors">
              <Send size={18} />
            </button>
          </div>
          <div className="text-xs text-gray-500 text-center mt-2">Role: {currentRole} | Tenant: {currentTenant}</div>
        </div>
      </div>
    </div>
  );
}
