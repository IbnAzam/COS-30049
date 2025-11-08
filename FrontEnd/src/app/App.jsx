// src/App.jsx
import { ThemeProvider, CssBaseline, createTheme } from '@mui/material';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import AppLayout from './AppLayout.jsx';
import Home from '../pages/Home.jsx';
import Scan from '../pages/Scan.jsx';
import Stats from '../pages/Stats.jsx';
import NotFound from '../pages/NotFound.jsx';

// optional custom palette
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: '#1976d2' },
    background: { default: '#a7adc8ff' }, // ok (8-digit hex with alpha)
  },
});

// sensible defaults for a “near real-time” dashboard
const qc = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: true,        // free freshness when tab refocuses
      refetchOnReconnect: true,
      refetchIntervalInBackground: false, // pause when tab not focused
      retry: 1,                           // quick fail for dev
      staleTime: 5_000,                   // treat data as fresh for 5s
      gcTime: 5 * 60_000,                 // cache for 5 min
    },
  },
});

export default function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <QueryClientProvider client={qc}>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<AppLayout />}>
              <Route index element={<Home />} />
              <Route path="scan" element={<Scan />} />
              <Route path="stats" element={<Stats />} />
              <Route path="*" element={<NotFound />} />
            </Route>
          </Routes>
        </BrowserRouter>

      </QueryClientProvider>
    </ThemeProvider>
  );
}
