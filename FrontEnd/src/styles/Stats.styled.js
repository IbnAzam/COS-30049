import { Box, styled } from '@mui/material';

export const StatsPage = styled(Box)({
    minHeight: '100vh',
    fontSize: 30,

    /* Two-column layout: left grows, right fixed width */
    display: 'grid',
    gridTemplateRows: 'auto 1fr',
    gridTemplateColumns: 'minmax(480px, 1fr) 320px',
    gap: 24,
    padding: '16px 24px 48px',

    '& h1': {
    gridColumn: '1 / -1',
    margin: 0,
    fontSize: '3rem',
    },


    /* Left column form */
    '& form': {
        display: 'flex',
        flexDirection: 'column',
        gap: 12,
    },

    '& textarea': {
        width: '100%',
        height: '300px',
        fontSize: '0.97rem',
        padding: '10px',
        boxSizing: 'border-box',
        resize: 'vertical',
        border: '2px solid #333',
        outline: 'none',
        overflowY: 'auto',
        scrollbarGutter: 'stable',
        backgroundColor: '#fff',
        borderRadius: '6px',
        '&:focus': { borderColor: '#3f51b5' },
    },

    '& .actions': {
        display: 'flex',
        gap: 10,
    },
});