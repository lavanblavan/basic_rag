import React, { useState } from 'react';
import axios from 'axios';

const PdfChatbot = () => {
    const [file, setFile] = useState(null);
    const [query, setQuery] = useState('');
    const [response, setResponse] = useState('');

    const handleFileChange = (event) => {
        setFile(event.target.files[0]);
    };

    const handleUpload = async () => {
        if (!file) return;
        const formData = new FormData();
        formData.append('file', file);
        await axios.post('http://localhost:5000/upload', formData);
        alert('File uploaded successfully!');
    };

    const handleQuery = async () => {
        const res = await axios.post('http://localhost:5000/query', { query });
        setResponse(res.data.response);
    };

    return (
        <div>
            <h1>PDF Chatbot</h1>
            <input type="file" onChange={handleFileChange} />
            <button onClick={handleUpload}>Upload PDF</button>
            <br /><br />
            <input type="text" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Ask a question..." />
            <button onClick={handleQuery}>Get Answer</button>
            <p><strong>Response:</strong> {response}</p>
        </div>
    );
};

export default PdfChatbot;