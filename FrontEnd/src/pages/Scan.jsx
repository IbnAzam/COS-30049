import { useState } from 'react';
import { ScanPage, Result } from '../styles/Scan.styled';
import { predictText } from '../api/predict';

function Scan() {
const [text, setText] = useState('');
const [result, setResult] = useState(null);     // { label, probability, confidence_pct }
const [loading, setLoading] = useState(false);
const [error, setError] = useState('');

const handleScan = async () => {
    if (!text.trim()) return;
    setLoading(true); setError('');
    try {
    const data = await predictText(text);       // uses /api/predict via Vite proxy
    setResult(data);
    console.log(data);
    } catch (e) {
    setError(e?.response?.data?.detail || e?.message || 'Failed to reach API');
    } finally {
    setLoading(false);
    }
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
        <button type="button" onClick={handleScan} disabled={loading || !text.trim()}>
            {loading ? 'Scanning…' : 'Scan'}
        </button>

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
            disabled={loading}
        >
            Upload
        </button>
        </div>

        {error && <p style={{ color: 'crimson', marginTop: 8 }}>{error}</p>}
    </form>

    {/* Right column */}
    <Result>
        <div className="card">
            <div className="circle">
                {result ? Math.round(result.probability * 100) : 0}
                <div className="percent">%</div>
            </div>

            <div className="label">
                {result ? "Likely to be Spam": 'Spam Score'}
            </div>

            <div className="verdict">
                {result
                    ? result.probability > 0.5
                    ? "Verdict: Spam detected 🚨"
                    : "Verdict: Clean (Ham) ✅"
                    : ""}
            </div>

            
        
        </div>
    </Result>
    </ScanPage>
);
}

export default Scan;
