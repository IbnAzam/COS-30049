
import { Title, MiniStats } from '../styles/Home.styled';

function Home(){ 
    
    return (
        <>
            <Title><h1>Welcome Back, User!</h1> </Title>

            <MiniStats>
                <h3>Statistics</h3>
                
                <ul>
                    <li>Total Emails Scanned: </li>
                    <li>Total Detected Spam: </li>
                    <li>Last Detected Spam: </li>
                </ul>
            </MiniStats>
        </>
    ); 
}

export default Home;
