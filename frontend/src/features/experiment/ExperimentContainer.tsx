import React, { useEffect } from 'react';
import { useExperiment } from './hooks/useExperiment';
import { IntroScreen } from './components/IntroScreen';
import { Questionnaire } from './components/Questionnaire';
import { DemographicsForm } from './components/DemographicsForm';
import { LearningScreen } from './components/LearningScreen';
import { ExperimentLayout } from './components/ExperimentLayout';
import { Icon } from '../../components/ui/Icons';

const ExperimentContainer: React.FC = () => {
    const { state, actions } = useExperiment();
    const { view, currentTrial, isLoading, error, feedback, session } = state;

    // Stylizacja body (tak jak w oryginale)
    useEffect(() => {
        document.body.style.backgroundColor = '#f8fafc';
        document.body.style.display = 'flex';
        document.body.style.alignItems = 'center';
        document.body.style.justifyContent = 'center';
        document.body.style.minHeight = '100vh';
        document.body.style.margin = '0';
        
        // Cleanup
        return () => {
            document.body.style.backgroundColor = '';
            document.body.style.display = '';
        };
    }, []);

    if (view === 'intro') {
        return <IntroScreen onStart={actions.startExperiment} />;
    }

    if (view === 'questionnaire') {
        return <Questionnaire onSubmit={actions.handleQuestSubmit} />;
    }

    if (view === 'demographics') {
        return <DemographicsForm onSubmit={actions.handleFinalSubmit} />;
    }

    if (view === 'done') {
        return (
            <div className="w-full max-w-xl bg-white rounded-[2.5rem] shadow-2xl text-center p-12 border-8 border-white mx-auto mt-12">
                <Icon.CheckCircle size={64} className="text-green-500 mx-auto mb-6" />
                <h1 className="text-4xl font-black text-slate-700 mb-4 tracking-tight">Experiment Completed!</h1>
                <p className="text-slate-500 mb-8">Thank you for your participation. Your data has been saved.</p>
                <button onClick={() => window.location.reload()} className="px-10 py-4 bg-blue-600 text-white rounded-2xl font-bold shadow-lg hover:bg-blue-700 transition-all active:scale-95">Start New Session</button>
            </div>
        );
    }

    // Widok eksperymentu (Learning lub Testing)
    return (
        <div className="flex flex-col items-center gap-6 w-full max-w-7xl px-4 py-8">
            {/* Globalny Error Handler */}
            {error && (
                <div className="fixed top-4 left-1/2 -translate-x-1/2 max-w-sm px-6 py-4 bg-red-600 text-white text-xs font-black rounded-full shadow-2xl flex items-center justify-center gap-2 z-[100] animate-bounce">
                    <Icon.Info size={20} /> {error}
                </div>
            )}

            {currentTrial?.task_type === 'learning_step' ? (
                 <LearningScreen 
                    data={currentTrial.payload} 
                    onNext={() => actions.submitAnswer('next_step')} 
                />
            ) : (
                <ExperimentLayout 
                    trial={currentTrial}
                    feedback={feedback}
                    isLoading={isLoading}
                    onSubmit={actions.submitAnswer}
                    onSkip={actions.skipToPhase}
                    onNextTrial={() => actions.fetchNextTrial(session.session_id)}
                />
            )}
        </div>
    );
};

export default ExperimentContainer;