import React, { useEffect } from 'react';
import { useExperiment } from './hooks/useExperiment';
import { IntroScreen } from './components/IntroScreen';
import { Questionnaire } from './components/Questionnaire';
import { DemographicsForm } from './components/DemographicsForm';
import { LearningScreen } from './components/LearningScreen';
import { ExperimentLayout } from './components/ExperimentLayout';
import { TutorPanel } from './components/TutorPanel';
import { Icon } from '../../components/ui/Icons';

const ExperimentContainer: React.FC = () => {
    const { state, actions } = useExperiment();

    // --- POPRAWKA 1: Destrukturyzacja 'language' ze stanu ---
    const { view, currentTrial, isLoading, error, feedback, nudge, session, language } = state;

    useEffect(() => {
        document.body.style.backgroundColor = '#f8fafc';
        document.body.style.display = 'flex';
        document.body.style.alignItems = 'center';
        document.body.style.justifyContent = 'center';
        document.body.style.minHeight = '100vh';
        document.body.style.margin = '0';
        return () => {
            document.body.style.backgroundColor = '';
            document.body.style.display = '';
        };
    }, []);

    if (view === 'intro') return <IntroScreen onStart={actions.startExperiment} />;
    if (view === 'questionnaire') return <Questionnaire onSubmit={actions.handleQuestSubmit} condition={session?.condition} />;
    if (view === 'demographics') return <DemographicsForm onSubmit={actions.handleFinalSubmit} />;

    if (view === 'done') {
        return (
            <div className="w-full max-w-xl bg-white rounded-[2.5rem] shadow-2xl text-center p-12 border-8 border-white mx-auto mt-12 animate-in fade-in zoom-in duration-500">
                <Icon.CheckCircle size={80} className="text-green-500 mx-auto mb-6" />
                <h1 className="text-4xl font-black text-slate-800 mb-4 tracking-tight">Experiment Completed!</h1>
                <p className="text-slate-500 mb-8 text-lg">Thank you for your participation. Your data has been successfully saved.</p>
                <button onClick={() => window.location.reload()} className="px-10 py-4 bg-blue-600 text-white rounded-2xl font-bold shadow-lg hover:bg-blue-700 transition-all active:scale-95">Start New Session</button>
            </div>
        );
    }

    // --- LOGIKA GŁÓWNA EKSPERYMENTU ---

    // Tutor widoczny dla wszystkich w fazie Learning, ALE TYLKO DLA GRUPY B
    const showTutor = currentTrial?.phase === 'learning' && session?.condition === 'B';

    // DEBUG LOGS
    console.log("DEBUG SWITCHER:", {
        phase: currentTrial?.phase,
        condition: session?.condition,
        showTutor,
        showSwitcher: currentTrial?.phase === 'learning' && !showTutor
    });

    const taskContext = {
        prompt: currentTrial?.payload?.question_context || currentTrial?.payload?.title || "Word learning",
        userAnswer: "",
        expectedAnswer: currentTrial?.payload?.german_word,
        exerciseType: currentTrial?.payload?.interaction_type
    };

    return (
        <div className="flex flex-col items-center gap-4 w-full max-w-[90rem] px-4 py-2">
            {error && (
                <div className="fixed top-4 left-1/2 -translate-x-1/2 max-w-sm px-6 py-4 bg-red-600 text-white text-xs font-black rounded-full shadow-2xl flex items-center justify-center gap-2 z-[100] animate-bounce">
                    <Icon.Info size={20} /> {error}
                </div>
            )}

            {currentTrial?.task_type === 'learning_step' ? (

                <div className="flex flex-col lg:flex-row gap-6 w-full justify-center items-start">

                    {/* LEWA STRONA: Ekran Zadania */}
                    <div className="flex-1 w-full min-w-0 transition-all duration-500">
                        <LearningScreen
                            data={currentTrial.payload}
                            onNext={() => actions.submitAnswer('next_step')}
                            // --- POPRAWKA 2: Przekazanie języka do LearningScreen ---
                            language={language}
                            // LANGUAGE SWITCHER FOR STATIC GROUP (Only in Learning Phase, Condition A)
                            showLanguageSwitcher={currentTrial?.phase === 'learning' && !showTutor}
                            onLanguageChange={() => actions.setLanguage(prev => prev === 'de' ? 'en' : 'de')}
                        />
                    </div>

                    {/* PRAWA STRONA: Panel Tutora */}
                    {showTutor && (
                        <div className="w-full lg:w-[350px] shrink-0 animate-in slide-in-from-right-10 duration-700">
                            <TutorPanel
                                isVisible={true}
                                currentTaskContext={taskContext}
                                nudge={nudge}
                                sessionId={session.session_id} // PASS SESSION ID
                                // --- POPRAWKA 3: Przekazanie języka i funkcji zmiany do Tutora ---
                                language={language}
                                onLanguageChange={() => actions.setLanguage(prev => prev === 'de' ? 'en' : 'de')}
                            />
                        </div>
                    )}
                </div>

            ) : (
                // --- FAZA TESTOWA ---
                <ExperimentLayout
                    trial={currentTrial}
                    feedback={feedback}
                    isLoading={isLoading}
                    onSubmit={actions.submitAnswer}
                    onSkip={actions.skipToPhase}
                    onNextTrial={() => actions.fetchNextTrial(session.session_id)}

                    // LANGUAGE SWITCHER FOR STATIC GROUP (Only in Learning Phase)
                    showLanguageSwitcher={currentTrial?.phase === 'learning' && !showTutor}
                    language={language}
                    onLanguageChange={() => actions.setLanguage(prev => prev === 'de' ? 'en' : 'de')}
                />
            )}
        </div>
    );
};

export default ExperimentContainer;