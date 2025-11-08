import { StatsPage } from '../styles/Stats.styled';
import StackedBar7d from '../dataVis/StackedBar7d';
import PieSpamHam from "../dataVis/PieSpamHam";
import ProbabilityHistogram from "../dataVis/ProbabilityHistogram";


export default function Stats() {
    return (
        <StatsPage>
            <h1>Stats</h1>

            <div className="fullwidth">
                <h3>Spam vs Ham (Last 7 Days)</h3>
                <StackedBar7d width={1200} height={250} />
            </div>

            <div style={{ maxWidth: 560 }}>
                <h3>Total Ratio</h3>
                <PieSpamHam width={520} height={300} />
            </div>

            <div style={{ maxWidth: 560 }}>
                <h3>Confidence</h3>
                <ProbabilityHistogram width={600} height={400} />
            </div>
        
        </StatsPage>
    );
}
