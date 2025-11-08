import { Box, styled } from '@mui/material';


export const ScanPage = styled(Box)({
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

export const Result = styled(Box)({
    position: 'sticky',
    top: 80,
    alignSelf: 'start',

    '.card': {
        display: 'grid',
        gridTemplateColumns: 'auto 1fr', // circle fits; label gets the rest
        gridAutoRows: 'auto',            // rows grow to content
        gap: 16,
        padding: 16,
        border: '1px solid #d0d0d0',
        borderRadius: 12,
        background: '#fff',
        alignItems: 'center',
    },

    '.circle': {
        gridColumn: '1',
        gridRow: '1',
        flexShrink: 0,
        width: 100,
        height: 100,
        borderRadius: '50%',
        border: '4px solid #3f51b5',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '1.3em',
        lineHeight: 1,
    },

    '.percent': { fontSize: '0.5em', marginTop: 5 },

    '.label': {
        gridColumn: '2',
        gridRow: '1',
        alignSelf: 'center',
        fontSize: '2.1rem',
        lineHeight: 1.2,
    },

    '.verdict': {
        gridColumn: '1 / -1',  // span both columns
        gridRow: '2',
        marginTop: 8,
        fontSize: '1.2rem',
        color: '#555',
        // optional polish:
        display: 'flex',
        alignItems: 'center',
        gap: 8,
    },
});
