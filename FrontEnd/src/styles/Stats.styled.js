import { Box, styled } from '@mui/material';

export const StatsPage = styled(Box)({
  minHeight: '100vh',
  fontSize: 30,

  display: 'grid',
  gridTemplateRows: 'auto 1fr',
  gridTemplateColumns: 'minmax(480px, 1fr) 560px',
  gap: 24,
  padding: '16px 24px 48px',

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
