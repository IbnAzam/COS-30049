import { Box, styled } from '@mui/material';

export const StatsPage = styled(Box)({
  minHeight: '100vh',
  fontSize: 30,

  display: 'grid',
  gridTemplateRows: 'auto 1fr',
  gridTemplateColumns: 'minmax(520px, 1fr) minmax(420px, 1fr)',
  gap: 24,
  '@media (max-width: 1200px)': {
    gridTemplateColumns: '1fr',   // stack charts when space is limited
  },

  '& h1': {
    gridColumn: '1 / -1',
    margin: 0,
    fontSize: '3rem',
  },

  // 👇 helper class to let a section span both columns
  '& .fullwidth': {
    gridColumn: '1 / -1',
  },

  '.slice:focus': {
  outline: '1px solid rgba(0, 0, 0, 0.2)',
  outlineOffset: '2px',
},

});
