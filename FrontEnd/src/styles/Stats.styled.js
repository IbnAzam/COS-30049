// Stats.styled.js
import { Box, styled } from '@mui/material';

export const StatsPage = styled(Box)({
  minHeight: '100vh',
  fontSize: 30,

  display: 'grid',
  gridTemplateRows: 'auto auto',
  gridTemplateColumns: '1fr 1fr 1fr',   // two equal columns
  gap: 24,
  padding: '16px 24px 48px',

  '& h1': {
    gridColumn: '1 / -1',           // title across both columns
    margin: 0,
    fontSize: '3rem',
  },

  // generic card
  '& .card': {
    background: '#fff',
    border: '1px solid #d0d0d0',
    borderRadius: 12,
    padding: 16,
  },

  // first row: make it full width
  '& .fullwidth': {
    gridColumn: '1 / -1',           // span both columns
  },

    '& .span-2': {
    gridColumn: 'span 2',
  },

  // make embedded SVGs fill cards neatly
  '& .card svg': { display: 'block', width: '100%', height: 'auto' },

  '@media (max-width: 1100px)': {
    gridTemplateColumns: '1fr',     // stack when narrow
  },
});
