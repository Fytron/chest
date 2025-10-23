'use client';

import React, { useState } from 'react';

export default function DocumentQA() {
  const [files, setFiles] = useState<File[]>([]);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [sources, setSources] = useState<{source: string, content: string}[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setFiles(Array.from(event.target.files));
    }
  };

  const uploadDocuments = async () => {
    if (files.length === 0) {
      alert('Please select files to upload');
      return;
    }

    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });

    try {
      setIsLoading(true);
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const data = await response.json();
      alert(`Uploaded ${data.document_count} documents`);
    } catch (error) {
      console.error('Upload failed', error);
      alert('Document upload failed');
    } finally {
      setIsLoading(false);
    }
  };

  const askQuestion = async () => {
    if (!question) {
      alert('Please enter a question');
      return;
    }

    try {
      setIsLoading(true);
      const response = await fetch('http://localhost:8000/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: question })
      });

      if (!response.ok) {
        throw new Error('Question failed');
      }

      const data = await response.json();
      setAnswer(data.answer);
      setSources(data.sources);
    } catch (error) {
      console.error('Question failed', error);
      alert('Failed to get an answer');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Document Q&A</h1>
      
      {/* File Upload Section */}
      <div className="mb-4">
        <input 
          type="file" 
          multiple 
          onChange={handleFileUpload} 
          className="mb-2"
          accept=".pdf,.txt,.docx"
        />
        <button 
          onClick={uploadDocuments}
          disabled={isLoading}
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
        >
          {isLoading ? 'Uploading...' : 'Upload Documents'}
        </button>
      </div>

      {/* Question Asking Section */}
      <div className="mt-4">
        <textarea 
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Enter your question about the documents"
          className="w-full p-2 border rounded mb-2"
          rows={3}
        />
        <button 
          onClick={askQuestion}
          disabled={isLoading}
          className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
        >
          {isLoading ? 'Processing...' : 'Ask Question'}
        </button>
      </div>

      {/* Answer Display Section */}
      {answer && (
        <div className="mt-4 p-4 bg-gray-100 rounded">
          <h2 className="font-bold mb-2">Answer:</h2>
          <p>{answer}</p>
          
          <h3 className="font-bold mt-4 mb-2">Sources:</h3>
          {sources.map((source, index) => (
            <div key={index} className="mb-2 p-2 bg-white rounded">
              <p className="font-semibold">Source: {source.source}</p>
              <p className="text-sm">{source.content}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
