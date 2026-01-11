import ExperimentContainer from './features/experiment/ExperimentContainer';
import { AdminPanel } from './features/experiment/components/AdminPanel';

function App() {
  const path = window.location.pathname;

  if (path === '/admin') {
    return <AdminPanel />;
  }

  return (
    <ExperimentContainer />
  );
}

export default App;