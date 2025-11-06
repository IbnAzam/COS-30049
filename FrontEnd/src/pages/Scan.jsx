import { useState } from 'react';
import { ScanPage, Result } from '../styles/Scan.styled';

function Scan() {
const [text, setText] = useState('');

const handleScan = async () => {
    const res = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
    });
    const result = await res.json();
    console.log(result);
};

const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const content = await file.text();
    setText(content);
};

return (
    <ScanPage>
    <h1>Scan</h1>

    {/* Left column */}
    <form onSubmit={(e) => e.preventDefault()}>
        <textarea
        placeholder="Paste or type text here..."
        value={text}
        onChange={(e) => setText(e.target.value)}
        />
        <div className="actions">
        <button type="button" onClick={handleScan}>Scan</button>

        {/* hidden file input, triggered by Upload */}
        <input
            id="fileInput"
            type="file"
            accept=".txt"
            style={{ display: 'none' }}
            onChange={handleUpload}
        />
        <button
            type="button"
            onClick={() => document.getElementById('fileInput').click()}
        >
            Upload
        </button>
        </div>
    </form>

    {/* Right column */}
    <Result>
        <div className="card">
        <div className="circle">0%</div>
        <div className="label">Spam score</div>
        </div>
    </Result>
    </ScanPage>
);
}

export default Scan;
