import React, { useEffect } from 'react';
import { useExperiment } from './hooks/useExperiment';
import { IntroScreen } from './components/IntroScreen';
import { Questionnaire } from './components/Questionnaire';
import { DemographicsForm } from './components/DemographicsForm';
import { LearningScreen } from './components/LearningScreen';
import { ExperimentLayout } from './components/ExperimentLayout';
import { TutorPanel } from './components/TutorPanel';
import { LandingPage } from './components/LandingPage';
import { Icon } from '../../components/ui/Icons';

const ExperimentContainer: React.FC = () => {
    const { state, actions } = useExperiment();

    // --- AUTH STATE ---
    const [accessCode, setAccessCode] = React.useState<string | null>(null);
    const [assignedGroup, setAssignedGroup] = React.useState<string | null>(null);
    const [authStep, setAuthStep] = React.useState(true); // true = showing LandingPage

    // --- POPRAWKA 1: Destrukturyzacja 'language' ze stanu ---
    const { view, currentTrial, isLoading, error, feedback, nudge, session, language, hasUsedTutor } = state;

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

    // --- FIX: Restore view if session is loaded from storage ---
    useEffect(() => {
        if (state.session && authStep) {
            setAuthStep(false);
            // setAccessCode("RESTORED"); // REMOVED: Leave as null so handleFinalSubmit uses localStorage
            setAssignedGroup(state.session.condition);
        }
    }, [state.session, authStep]);

    if (authStep) {
        return (
            <LandingPage onSuccess={(code, group) => {
                setAccessCode(code);
                localStorage.setItem('experiment_access_code', code); // FIX: Persist code for refresh
                setAssignedGroup(group);
                setAuthStep(false);
            }} />
        );
    }

    if (view === 'intro') {
        return (
            <IntroScreen
                onStart={(grp) => actions.startExperiment(grp as 'A' | 'B')} // Cast for strict type in hook, though hook accepts string in practice if updated
                assignedGroup={assignedGroup as 'A' | 'B'}
            />
        );
    }
    if (view === 'questionnaire') return <Questionnaire onSubmit={actions.handleQuestSubmit} condition={session?.condition} tutorUsed={hasUsedTutor} />;
    if (view === 'demographics') {
        return (
            <DemographicsForm onSubmit={(data) => actions.handleFinalSubmit({ ...data, access_code: accessCode })} />
        );
    }

    if (view === 'done') {
        return (
            <div className="w-full max-w-xl bg-white rounded-[2.5rem] shadow-2xl text-center p-12 border-8 border-white mx-auto mt-12 animate-in fade-in zoom-in duration-500">
                <Icon.CheckCircle size={80} className="text-green-500 mx-auto mb-6" />
                <h1 className="text-4xl font-black text-slate-800 mb-4 tracking-tight">Experiment Completed!</h1>
                <p className="text-slate-500 mb-8 text-lg">Thank you for your participation. Your data has been successfully saved.</p>
            </div>
        );
    }

    // --- LOGIKA GŁÓWNA EKSPERYMENTU ---

    // Tutor widoczny dla wszystkich w fazie Learning, ALE TYLKO DLA GRUPY B
    const showTutor = currentTrial?.phase === 'learning' && session?.condition === 'B';



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
                            onNext={(result) => actions.submitAnswer('next_step', result)}
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
                                language={language}
                                onLanguageChange={() => actions.setLanguage(prev => prev === 'de' ? 'en' : 'de')}
                                // PERSISTENCE
                                initialHistory={currentTrial?.tutor_state?.history}
                                initialPromptCount={currentTrial?.tutor_state?.prompt_count}
                                onInteraction={actions.markTutorUsed}
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
                    loadingText={
                        (isLoading && session?.condition === 'B' && currentTrial?.phase === 'pre-test')
                            ? "Generating personalized tasks..."
                            : "Loading..."
                    }
                    onSubmit={actions.submitAnswer}
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