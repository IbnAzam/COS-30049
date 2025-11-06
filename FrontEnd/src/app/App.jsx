// src/App.jsx
import { ThemeProvider, CssBaseline, createTheme } from '@mui/material';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
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
    background: { default: '#a7adc8ff' },
  },
});

export default function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
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
    </ThemeProvider>
  );
}
