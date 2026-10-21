'use client';

import { useState } from 'react';
import { Upload, FileText, ArrowLeft, CheckCircle } from 'lucide-react';
import Link from 'next/link';
import { api } from '@/lib/api';
import { toast } from 'sonner';

export default function DocumentsPage() {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [uploadedFiles, setUploadedFiles] = useState<any[]>([]);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    setUploading(true);
    setProgress(0);

    try {
      const interval = setInterval(() => {
        setProgress((p) => Math.min(p + 10, 90));
      }, 200);

      const res = await api.post('/documents/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      clearInterval(interval);
      setProgress(100);
      setUploadedFiles((prev) => [res.data, ...prev]);
      toast.success('Document uploaded and indexed successfully!');
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Upload failed');
    } finally {
      setUploading(false);
      setTimeout(() => setProgress(0), 1000);
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white">
      <div className="border-b border-gray-800 p-4">
        <div className="max-w-6xl mx-auto flex items-center gap-4">
          <Link href="/" className="p-2 hover:bg-[#1a1a1a] rounded-lg">
            <ArrowLeft size={20} />
          </Link>
          <h1 className="text-xl font-semibold">Knowledge Base</h1>
        </div>
      </div>

      <div className="max-w-6xl mx-auto p-6 space-y-6">
        <div className="border-2 border-dashed border-gray-700 rounded-lg p-12 text-center hover:border-blue-500 transition-colors">
          <input
            type="file"
            onChange={handleUpload}
            accept=".pdf,.docx,.doc,.pptx,.ppt,.txt,.json,.xml,.md"
            className="hidden"
            id="file-upload"
            disabled={uploading}
          />
          <label htmlFor="file-upload" className="cursor-pointer space-y-4 block">
            <Upload className="w-12 h-12 mx-auto text-gray-400" />
            <div>
              <p className="text-lg font-medium">Drop files here or click to upload</p>
              <p className="text-sm text-gray-400 mt-1">Supports PDF, DOCX, PPTX, TXT, JSON, XML</p>
            </div>
          </label>
          {uploading && (
            <div className="mt-4 max-w-md mx-auto">
              <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                <div className="h-full bg-blue-600 transition-all duration-300" style={{ width: `${progress}%` }} />
              </div>
              <p className="text-sm text-gray-400 mt-2">Uploading and indexing...</p>
            </div>
          )}
        </div>

        <div className="space-y-4">
          <h2 className="text-lg font-semibold">Uploaded Documents</h2>
          {uploadedFiles.length === 0 ? (
            <div className="text-center py-12 text-gray-400">No documents uploaded yet</div>
          ) : (
            <div className="grid gap-3">
              {uploadedFiles.map((doc: any, idx) => (
                <div key={idx} className="flex items-center gap-3 p-4 bg-[#111111] border border-gray-800 rounded-lg">
                  <FileText className="text-blue-400" size={24} />
                  <div className="flex-1">
                    <p className="font-medium">{doc.document?.title || doc.title}</p>
                    <p className="text-sm text-gray-400">{doc.chunks_created} chunks indexed</p>
                  </div>
                  <CheckCircle className="text-green-400" size={20} />
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
