import { useEffect, useState } from "react";
import axios from "axios";
import { Title, MiniStats } from "../styles/Home.styled";
import { fmtMelTime } from "../utils/time";

const API = import.meta.env.VITE_API_BASE || "http://localhost:8000";

function Home() {
  const [summary, setSummary] = useState(null);
  const [lastSpam, setLastSpam] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    (async () => {
      try {
        const [s, ls] = await Promise.all([
          axios.get(`${API}/api/stats/summary`),
          axios.get(`${API}/api/stats/latest-spam`),
        ]);
        setSummary(s.data);
        setLastSpam(ls.data?.created_at || null); // backend still UTC
      } catch (e) {
        setError(e?.response?.data?.detail || e.message);
      }
    })();
  }, []);

  const total = summary?.total_scans ?? 0;
  const spamCount = summary?.by_label?.find((x) => x.label === "Spam")?.c ?? 0;

  return (
    <>
      <Title><h1>Welcome Back, User!</h1></Title>

      <MiniStats>
        <h3>Statistics</h3>
        {error && <p style={{ color: "crimson" }}>{error}</p>}

        {!summary ? (
          <p>Loading stats...</p>
        ) : (
          <ul>
            <li>Total Emails Scanned: <b>{total}</b></li>
            <li>Total Detected Spam: <b>{spamCount}</b></li>
            <li>Last Detected Spam: <b>{fmtMelTime(lastSpam)}</b></li>
          </ul>
        )}
      </MiniStats>
    </>
  );
}

export default Home;
