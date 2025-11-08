import { useQuery } from "@tanstack/react-query";
import axios from "axios";
import { Title, MiniStats } from "../styles/Home.styled";
import { fmtMelTime } from "../utils/time";

const API = import.meta.env.VITE_API_BASE || "http://localhost:8000";

// ---- Fetchers ----
async function fetchSummary() {
  const res = await axios.get(`${API}/api/stats/summary`);
  return res.data;
}

async function fetchLastSpam() {
  const res = await axios.get(`${API}/api/stats/latest-spam`);
  return res.data?.created_at || null;
}

function Home() {
  // ✅ parallel React Query calls
  const summaryQuery = useQuery({
    queryKey: ["summary"],
    queryFn: fetchSummary,
    refetchInterval: 10000,
    refetchOnWindowFocus: true,
    staleTime: 5000,
  });

  const lastSpamQuery = useQuery({
    queryKey: ["latest-spam"],
    queryFn: fetchLastSpam,
    refetchInterval: 10000,
    refetchOnWindowFocus: true,
    staleTime: 5000,
  });

  const summary = summaryQuery.data;
  const lastSpam = lastSpamQuery.data;
  const isLoading = summaryQuery.isLoading || lastSpamQuery.isLoading;
  const error = summaryQuery.error || lastSpamQuery.error;

  const total = summary?.total_scans ?? 0;
  const spamCount = summary?.by_label?.find((x) => x.label === "Spam")?.c ?? 0;

  return (
    <>
      <Title><h1>Welcome Back, User!</h1></Title>

      <MiniStats>
        <h3>Statistics</h3>

        {error && (
          <p style={{ color: "crimson" }}>
            {error.message || "Failed to load stats"}
          </p>
        )}

        {isLoading ? (
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
