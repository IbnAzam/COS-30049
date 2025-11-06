// app/AppLayout.jsx
import { Outlet } from 'react-router-dom';
import Nav from '../nav/Nav.jsx';
import { Root, Sidebar, Main } from '../styles/AppLayout.styled';

export default function AppLayout() {
    return (
        <Root>
            <Sidebar><Nav/></Sidebar>
            <Main><Outlet /></Main>
        </Root>
    );
}
