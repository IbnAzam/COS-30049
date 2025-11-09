

// app/AppLayout.styled.js
import { Box, hslToRgb, styled } from '@mui/material';

export const Root = styled(Box)({
    display: 'flex',
    minHeight: '100vh',
});

export const Sidebar = styled(Box)(({ theme }) => ({
    width: 280,
    flexShrink: 0,
    backgroundColor: 'hsla(222, 45%, 16%, 1.00)',
    borderRight: `3px solid ${theme.palette.divider}`,

    // Target list items inside
    '& ul': {
        listStyle: 'none',
        padding: 5,
        marginLeft: 50,
        marginRight: 50,
        marginTop: 60,
    },
    '& li': {
        borderRadius: 10,
        cursor: 'pointer',
        marginBottom: 25,
        textAlign: 'center',
        border: '2px solid #fff',
        transition: 'background-color 0.3s',
        '&:hover': {
        backgroundColor: theme.palette.grey[800],
        },
    },
    '& a': {
        textDecoration: 'none',
        color: 'white',
        display: 'block',
        padding: 5,
        fontSize: 20,
        transition: 'color 0.3s, background-color 0.3s',
        borderRadius: 10,

        '&.active': {
            backgroundColor: '#444',  // grey background for active tab
            color: '#bbb',            // faded text
            borderRadius: 10,
            pointerEvents: 'none',    // disables clicking again
        },
    },

}));

export const Main = styled(Box)({   
    flexGrow: 1,
    padding: 20,
    position: 'relative'
});
