import { useQuery } from "@tanstack/react-query";
import axios from "axios";
import { Title, MiniStats } from "../styles/Home.styled";
import { fmtMelTime } from "../utils/time";

const API = import.meta.env.VITE_API_BASE || "http://localhost:8000";

// ---- Fetchers ----
async function fetchSummary() {
  const { data } = await axios.get(`${API}/api/stats/summary`);
  return data;
}
async function fetchLastSpam() {
  const { data } = await axios.get(`${API}/api/stats/latest-spam`);
  return data?.created_at || null;
}

function Home() {
  // Keep previous values during background refetches
  const summaryQuery = useQuery({
    queryKey: ["summary"],
    queryFn: fetchSummary,
    refetchInterval: 10000,
    refetchOnWindowFocus: true,
    staleTime: 5000,
    placeholderData: (prev) => prev, // prevents momentary "empty"
    select: (d) => {
      if (!d) return d;
      const byLabel = Array.isArray(d.by_label) ? d.by_label : [];
      // normalize map for safety
      const map = Object.fromEntries(
        byLabel.map((x) => [x.label, Number(x.c ?? 0)])
      );
      return {
        total_scans: Number(d.total_scans ?? 0),
        by_label: byLabel,
        by_label_map: { Ham: map.Ham ?? 0, Spam: map.Spam ?? 0 },
      };
    },
  });

  const lastSpamQuery = useQuery({
    queryKey: ["latest-spam"],
    queryFn: fetchLastSpam,
    refetchInterval: 10000,
    refetchOnWindowFocus: true,
    staleTime: 5000,
    placeholderData: (prev) => prev, // keep previous while refetching
  });

  // Only render stats when both are ready
  const summaryReady = summaryQuery.status === "success" && summaryQuery.data;
  const lastSpamReady = lastSpamQuery.status === "success";

  const error = summaryQuery.error || lastSpamQuery.error;

  const total = summaryReady ? summaryQuery.data.total_scans : undefined;
  const spamCount = summaryReady ? summaryQuery.data.by_label_map.Spam : undefined;
  const lastSpam = lastSpamReady ? lastSpamQuery.data : null;

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

        {!summaryReady || !lastSpamReady ? (
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
