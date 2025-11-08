// useStatsData.js
import { useQuery } from '@tanstack/react-query';

const API = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

export const useTimeseries7d = () =>
  useQuery({
    queryKey: ['timeseries', 'day', 7],
    queryFn: async () => {
      const r = await fetch(`${API}/api/stats/timeseries?bucket=day&days=7&tz=Australia/Melbourne`);
      return r.json();
    },
    refetchInterval: 10000,           // 10s
    refetchOnWindowFocus: true,
    refetchIntervalInBackground: false,
    staleTime: 5000,
  });

export const useSummary = () =>
  useQuery({
    queryKey: ['summary'],
    queryFn: async () => (await fetch(`${API}/api/stats/summary`)).json(),
    refetchInterval: 12000,           // stagger intervals a bit
    refetchOnWindowFocus: true,
    staleTime: 5000,
  });

export const useDistribution = (bins = 20) =>
  useQuery({
    queryKey: ['distribution', bins],
    queryFn: async () => (await fetch(`${API}/api/stats/distribution?bins=${bins}`)).json(),
    refetchInterval: 15000,
    refetchOnWindowFocus: true,
    staleTime: 5000,
  });
