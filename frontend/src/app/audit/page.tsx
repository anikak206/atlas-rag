'use client';

import { useState, useEffect } from 'react';
import { ArrowLeft, Clock, User, FileText, CheckCircle, XCircle } from 'lucide-react';
import Link from 'next/link';
import { api } from '@/lib/api';
import { format } from 'date-fns';

export default function AuditPage() {
  const [audits, setAudits] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAudits = async () => {
      try {
        const res = await api.get('/audit?limit=50');
        setAudits(res.data.items || []);
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    };
    fetchAudits();
  }, []);

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white">
      <div className="border-b border-gray-800 p-4">
        <div className="max-w-7xl mx-auto flex items-center gap-4">
          <Link href="/" className="p-2 hover:bg-[#1a1a1a] rounded-lg">
            <ArrowLeft size={20} />
          </Link>
          <h1 className="text-xl font-semibold">Audit Dashboard</h1>
        </div>
      </div>

      <div className="max-w-7xl mx-auto p-6">
        <div className="bg-[#111111] border border-gray-800 rounded-lg overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-[#0a0a0a] border-b border-gray-800">
                <tr>
                  <th className="text-left p-4 text-sm font-medium text-gray-400">Status</th>
                  <th className="text-left p-4 text-sm font-medium text-gray-400">Query</th>
                  <th className="text-left p-4 text-sm font-medium text-gray-400">User</th>
                  <th className="text-left p-4 text-sm font-medium text-gray-400">Role</th>
                  <th className="text-left p-4 text-sm font-medium text-gray-400">Latency</th>
                  <th className="text-left p-4 text-sm font-medium text-gray-400">Tokens</th>
                  <th className="text-left p-4 text-sm font-medium text-gray-400">Chunks</th>
                  <th className="text-left p-4 text-sm font-medium text-gray-400">Time</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr><td colSpan={8} className="text-center p-8 text-gray-400">Loading...</td></tr>
                ) : audits.length === 0 ? (
                  <tr><td colSpan={8} className="text-center p-8 text-gray-400">No audit logs yet</td></tr>
                ) : (
                  audits.map((audit) => (
                    <tr key={audit.id} className="border-b border-gray-800 hover:bg-[#0a0a0a]/50">
                      <td className="p-4">
                        {audit.success ? (
                          <CheckCircle className="text-green-400" size={18} />
                        ) : (
                          <XCircle className="text-red-400" size={18} />
                        )}
                      </td>
                      <td className="p-4 max-w-xs">
                        <div className="truncate text-sm">{audit.query_text}</div>
                      </td>
                      <td className="p-4 text-sm">{audit.actor_email || 'N/A'}</td>
                      <td className="p-4">
                        <span className="px-2 py-1 bg-[#1a1a1a] rounded text-xs">{audit.actor_role || 'N/A'}</span>
                      </td>
                      <td className="p-4 text-sm text-gray-400">{audit.latency?.toFixed(0)}ms</td>
                      <td className="p-4 text-sm text-gray-400">{audit.tokens_used}</td>
                      <td className="p-4 text-sm text-gray-400">{audit.retrieved_chunk_count || 0}</td>
                      <td className="p-4 text-sm text-gray-400">
                        {audit.created_at ? format(new Date(audit.created_at), 'MMM dd HH:mm') : 'N/A'}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
