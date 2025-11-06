
import { Box, hslToRgb, styled } from '@mui/material';


export const Title = styled(Box)({
    fontSize: 30,
    textAlign: 'center',
    padding: 5,
    marginTop: 80,
});

export const MiniStats = styled(Box)(({ theme }) => ({

  position: 'absolute',
  left: 50,
  bottom: 100,
  fontSize: 20,
  padding: 5,

  '& ul': {
    listStyle: 'none',
    margin: 0,
    padding: 0,
  },
  '& li': {
    lineHeight: 1.4,
  },
}));
