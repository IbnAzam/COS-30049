import { Box, styled } from '@mui/material';


export const ScanPage = styled(Box)({
    minHeight: '100vh',
    fontSize: 30,

    /* Two-column layout: left grows, right fixed width */
    display: 'grid',
    gridTemplateColumns: 'minmax(480px, 1fr) 320px',
    alignItems: 'start',
    gap: 24,
    padding: '24px 24px 48px',
    
    '& h1': {
        gridColumn: '1 / -1',
        margin: 0,
        marginBottom: 12,
        fontSize: '2rem',
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

export const Result = styled(Box)({
    /* Right column panel */
    position: 'sticky',   // stays visible as you scroll
    top: 80,              // distance from top when sticky
    alignSelf: 'start',   // pin to top of its grid cell

    '.card': {
        padding: 16,
        borderRadius: 12,
        border: '1px solid #d0d0d0',
        background: '#fff',
        boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
        display: 'grid',
        placeItems: 'center',
        gap: 8,
    },

    '.circle': {
        width: 140,
        height: 140,
        borderRadius: '50%',
        border: '6px solid #3f51b5',
        display: 'grid',
        placeItems: 'center',
        fontSize: 24,
        fontWeight: 700,
    },

    '.label': {
        fontSize: 14,
        color: '#555',
    },
});
