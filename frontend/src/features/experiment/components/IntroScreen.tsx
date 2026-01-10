import { useState, useEffect } from 'react';
import { Icon } from '../../../components/ui/Icons';

interface IntroScreenProps {
    onStart: (cond: 'A' | 'B') => void;
}

export const IntroScreen = ({ onStart }: IntroScreenProps) => {
    // Stan zarządzający etapem: 'consent' (zgoda) lub 'instructions' (instrukcja)
    const [step, setStep] = useState<'consent' | 'instructions'>('consent');
    const [selectedGroup, setSelectedGroup] = useState<'A' | 'B' | null>(null);

    // --- LOGIKA CONSENT FORM ---
    const [showFullText, setShowFullText] = useState(false);
    const [consents, setConsents] = useState({
        isAdult: false,
        participate: false,
        webcam: false
    });

    const canProceed = consents.isAdult && consents.participate;

    const toggleConsent = (key: keyof typeof consents) => {
        setConsents(prev => ({ ...prev, [key]: !prev[key] }));
    };

    const handleConsentSubmit = (group: 'A' | 'B') => {
        setSelectedGroup(group);
        setStep('instructions');
        // Scroll to top
        window.scrollTo(0, 0);
    };

    // Scroll na górę przy załadowaniu
    useEffect(() => { window.scrollTo(0, 0); }, []);

    // --- WIDOK 1: FORMULARZ ZGODY ---
    if (step === 'consent') {
        return (
            <div className="w-full max-w-5xl bg-white rounded-[3rem] shadow-2xl p-12 border-8 border-white mx-auto text-slate-800 animate-in fade-in zoom-in duration-500 my-8">
                <h1 className="text-4xl font-black text-slate-900 mb-2 text-center">Consent Form</h1>
                <p className="text-center text-slate-500 mb-8 font-medium">Please read carefully before proceeding.</p>

                <div className="flex flex-col gap-6">
                    {/* SHORT VERSION */}
                    <div className="bg-slate-50 p-8 rounded-3xl border border-slate-200 text-lg space-y-6">
                        <div>
                            <h3 className="font-bold text-xl mb-2 flex items-center gap-2"><Icon.Info size={24} className="text-blue-600" /> Purpose</h3>
                            <p className="text-slate-600">This study evaluates a web-based German vocabulary learning system with different types of feedback.</p>
                        </div>

                        <div className="grid md:grid-cols-2 gap-6">
                            <div>
                                <h3 className="font-bold text-lg mb-2">What you will do</h3>
                                <ul className="list-disc ml-5 text-slate-600 space-y-1">
                                    <li>Complete a short vocabulary pre-test</li>
                                    <li>Complete a short learning session</li>
                                    <li>Complete a short post-test and questionnaire (Duration: ~15 minutes)</li>
                                </ul>
                            </div>
                            <div>
                                <h3 className="font-bold text-lg mb-2">Data collected</h3>
                                <ul className="list-disc ml-5 text-slate-600 space-y-1">
                                    <li>Task answers and response times</li>
                                    <li>Feedback condition (static or AI-based)</li>
                                    <li>Questionnaire responses</li>
                                    <li>(Optional) webcam-based gaze/attention metrics</li>
                                </ul>
                            </div>
                        </div>

                        <div className="grid md:grid-cols-2 gap-6">
                            <div>
                                <h3 className="font-bold text-lg mb-2">Webcam & AI</h3>
                                <ul className="list-disc ml-5 text-slate-600 space-y-1">
                                    <li>Webcam is used only to estimate gaze direction</li>
                                    <li><strong>No video recordings are stored</strong></li>
                                    <li>Some participants receive AI-generated feedback</li>
                                </ul>
                            </div>
                            <div>
                                <h3 className="font-bold text-lg mb-2">Rights & Legal</h3>
                                <ul className="list-disc ml-5 text-slate-600 space-y-1">
                                    <li>Participation is voluntary</li>
                                    <li>You can stop at any time without consequences</li>
                                    <li>Data processing is based on your consent (GDPR Art. 6(1)(a))</li>
                                </ul>
                            </div>
                        </div>
                    </div>

                    {/* FULL VERSION TOGGLE - CAŁA SZEROKOŚĆ */}
                    <div className="mb-2">
                        <button
                            onClick={() => setShowFullText(!showFullText)}
                            className="w-full py-4 rounded-2xl bg-indigo-50 text-indigo-700 border-2 border-indigo-100 font-bold hover:bg-indigo-100 hover:border-indigo-300 transition-all text-sm uppercase tracking-wide flex items-center justify-center gap-3 shadow-sm"
                        >
                            <Icon.Info size={20} />
                            {showFullText ? "Hide Full Participant Information" : "Show Full Participant Information & Detailed Consent"}
                        </button>
                    </div>

                    {/* FULL VERSION TEXT */}
                    {showFullText && (
                        <div className="bg-white p-6 rounded-3xl border-2 border-indigo-50 text-sm text-slate-600 h-96 overflow-y-auto shadow-inner leading-relaxed animate-in fade-in slide-in-from-top-4">
                            <h2 className="font-black text-lg text-slate-800 mb-2">Participant Information and Informed Consent (Full Version)</h2>

                            <p className="mb-2"><strong>Study Title:</strong> Human–AI Interaction in Vocabulary Learning</p>
                            <p className="mb-4"><strong>Institution:</strong> Technical University of Munich (TUM), Course: Human–AI Interaction</p>

                            <h3 className="font-bold text-slate-800 mt-4">1. Purpose of the Study</h3>
                            <p className="mb-2">You are invited to participate in a research study conducted as part of a university course project. The purpose of this study is to investigate how different types of feedback in a web-based vocabulary learning system influence learner engagement and short-term learning outcomes. Specifically, the study compares static (non-adaptive) feedback and adaptive, AI-generated feedback.</p>

                            <h3 className="font-bold text-slate-800 mt-4">2. Study Procedure</h3>
                            <p className="mb-2">If you agree to participate, you will complete the following steps: Read and agree to this consent form, complete a short vocabulary pre-test, complete a learning session involving image-based German vocabulary tasks, complete a post-test, and a short questionnaire. The study takes approximately 12–15 minutes.</p>

                            <h3 className="font-bold text-slate-800 mt-4">3. Tasks and Interaction</h3>
                            <p className="mb-2">During the learning tasks, you will be shown images and asked to answer simple vocabulary-related questions. Feedback may be presented as either static text or AI-generated explanations. The AI system is used solely to generate short feedback messages and does not make decisions about scoring.</p>

                            <h3 className="font-bold text-slate-800 mt-4">4. Webcam-Based Gaze / Attention Measurement</h3>
                            <p className="mb-2">This study may use your webcam to estimate gaze direction. Your webcam video is not recorded or stored. Only derived gaze information is collected. You may decline webcam access.</p>

                            <h3 className="font-bold text-slate-800 mt-4">5. Data Collected</h3>
                            <p className="mb-2">Responses to vocabulary tasks, response times, feedback condition, questionnaire responses, and estimated gaze metrics (if granted). The study does not intentionally collect sensitive personal data.</p>

                            <h3 className="font-bold text-slate-800 mt-4">6. Legal Basis (GDPR)</h3>
                            <p className="mb-2">Data processing is based on informed consent (GDPR Art. 6(1)(a)). You may withdraw your consent at any time without consequences.</p>

                            <h3 className="font-bold text-slate-800 mt-4">7. Data Storage & Retention</h3>
                            <p className="mb-2">Data will be stored on restricted systems. Participants are identified using a randomly generated ID. Data will be analyzed in aggregated form. Data will be retained until 31.03.2026.</p>

                            <h3 className="font-bold text-slate-800 mt-4">8. Contact</h3>
                            <p className="mb-2">Researchers: elham.tajalli@tum.de, alexander.brehmer@tum.de, ugur.alimoglu@tum.de, kacper.kolodziejczyk@tum.de</p>
                        </div>
                    )}

                    <hr className="border-slate-200" />

                    {/* CONSENT CHECKBOXES */}
                    <div className="space-y-4 px-4">
                        <label className="flex items-center gap-4 cursor-pointer p-4 rounded-xl hover:bg-slate-50 transition-colors border-2 border-transparent hover:border-slate-200">
                            <div className={`w-8 h-8 rounded-lg border-2 flex items-center justify-center transition-all ${consents.isAdult ? 'bg-blue-600 border-blue-600 text-white' : 'bg-white border-slate-300'}`}>
                                {consents.isAdult && <Icon.CheckCircle size={20} />}
                            </div>
                            <input type="checkbox" className="hidden" checked={consents.isAdult} onChange={() => toggleConsent('isAdult')} />
                            <span className="text-lg font-bold text-slate-700">I am at least 18 years old</span>
                        </label>

                        <label className="flex items-center gap-4 cursor-pointer p-4 rounded-xl hover:bg-slate-50 transition-colors border-2 border-transparent hover:border-slate-200">
                            <div className={`w-8 h-8 rounded-lg border-2 flex items-center justify-center transition-all ${consents.participate ? 'bg-blue-600 border-blue-600 text-white' : 'bg-white border-slate-300'}`}>
                                {consents.participate && <Icon.CheckCircle size={20} />}
                            </div>
                            <input type="checkbox" className="hidden" checked={consents.participate} onChange={() => toggleConsent('participate')} />
                            <span className="text-lg font-bold text-slate-700">I agree to participate in this study</span>
                        </label>

                        <label className="flex items-center gap-4 cursor-pointer p-4 rounded-xl hover:bg-slate-50 transition-colors border-2 border-transparent hover:border-slate-200">
                            <div className={`w-8 h-8 rounded-lg border-2 flex items-center justify-center transition-all ${consents.webcam ? 'bg-indigo-600 border-indigo-600 text-white' : 'bg-white border-slate-300'}`}>
                                {consents.webcam && <Icon.CheckCircle size={20} />}
                            </div>
                            <input type="checkbox" className="hidden" checked={consents.webcam} onChange={() => toggleConsent('webcam')} />
                            <span className="text-lg font-medium text-slate-600">I agree to webcam-based gaze data collection (optional)</span>
                        </label>
                    </div>

                    {/* ACTION BUTTONS */}
                    <div className="grid grid-cols-2 gap-6 mt-4">
                        <button
                            disabled={!canProceed}
                            onClick={() => handleConsentSubmit('A')}
                            className="py-5 bg-blue-600 text-white rounded-2xl font-bold text-xl hover:bg-blue-700 shadow-xl transition-all active:scale-95 disabled:bg-slate-300 disabled:shadow-none disabled:cursor-not-allowed"
                        >
                            {canProceed ? "Proceed (Group A)" : "Complete Consent"}
                        </button>
                        <button
                            disabled={!canProceed}
                            onClick={() => handleConsentSubmit('B')}
                            className="py-5 bg-purple-600 text-white rounded-2xl font-bold text-xl hover:bg-purple-700 shadow-xl transition-all active:scale-95 disabled:bg-slate-300 disabled:shadow-none disabled:cursor-not-allowed"
                        >
                            {canProceed ? "Proceed (Group B)" : "Complete Consent"}
                        </button>
                    </div>

                    {!canProceed && (
                        <p className="text-center text-red-500 font-bold text-sm">Please confirm your age and agreement to continue.</p>
                    )}
                </div>
            </div>
        );
    }

    // --- WIDOK 2: INSTRUKCJE ---
    return (
        <div className="w-full max-w-4xl bg-white rounded-[3rem] shadow-2xl p-12 border-8 border-white mx-auto text-slate-800 animate-in fade-in slide-in-from-right-8 duration-500 my-8">
            <h1 className="text-4xl font-black text-slate-900 mb-6 text-center">Study Instructions</h1>

            <div className="space-y-8">
                {/* 1. Experiment Flow */}
                <div className="bg-slate-50 p-6 rounded-3xl border border-slate-100">
                    <h3 className="font-bold text-xl mb-4 text-slate-700">Experiment Flow</h3>
                    <div className="flex items-center justify-between text-sm font-bold text-slate-500 uppercase tracking-widest px-4">
                        <span>1. Pre-Test</span>
                        <Icon.ArrowRight className="text-slate-300" />
                        <span className="text-indigo-600">2. Learning Phase</span>
                        <Icon.ArrowRight className="text-slate-300" />
                        <span>3. Post-Test</span>
                    </div>
                    <p className="mt-4 text-slate-600 text-lg">
                        {selectedGroup === 'A'
                            ? "In the learning phase, you will follow a structured path to study the vocabulary items with standard exercises."
                            : "In the learning phase, an AI tutor will analyze your mistakes and generate personalized exercises to help you improve."}
                    </p>
                </div>

                {selectedGroup === 'B' && (
                    <div className="bg-purple-50 p-6 rounded-3xl border border-purple-100 animate-in fade-in slide-in-from-bottom-4">
                        <div className="flex items-center gap-3 mb-3">
                            <Icon.Sparkles className="text-purple-600" size={24} />
                            <h3 className="font-bold text-xl text-slate-800">Meet Your AI Tutor</h3>
                        </div>
                        <p className="text-slate-600 leading-relaxed">
                            During the learning phase, you will see an <strong>AI Tutor panel</strong> on the right.
                            You can ask questions, request hints, or get explanations for your mistakes.
                            <br /><span className="italic text-purple-700 font-medium mt-1 block">Feel free to interact with it!</span>
                        </p>
                    </div>
                )}

                {/* 2. Critical Rules */}
                <div className="grid md:grid-cols-2 gap-6">
                    {/* Case Sensitivity */}
                    <div className="bg-yellow-50 p-6 rounded-3xl border border-yellow-100">
                        <div className="flex items-center gap-3 mb-3">
                            <Icon.Lightbulb className="text-amber-500" />
                            <h3 className="font-bold text-lg text-slate-800">Case Sensitive!</h3>
                        </div>
                        <p className="text-slate-600 mb-4">German nouns are always capitalized. Pay attention to your spelling.</p>
                        <div className="bg-white p-3 rounded-xl border border-yellow-200 flex justify-around text-lg">
                            <span className="text-green-600 font-bold flex items-center gap-1"><Icon.CheckCircle size={18} /> Tisch</span>
                            <span className="text-red-400 line-through decoration-2 decoration-red-400 opacity-60">tisch</span>
                        </div>
                    </div>

                    {/* Article Colors */}
                    <div className="bg-indigo-50 p-6 rounded-3xl border border-indigo-100">
                        <div className="flex items-center gap-3 mb-3">
                            <Icon.Info className="text-indigo-500" />
                            <h3 className="font-bold text-lg text-slate-800">Color Codes</h3>
                        </div>
                        <p className="text-slate-600 mb-4">We use colors to help you remember articles (Der/Die/Das).</p>
                        <div className="flex gap-2">
                            <span className="flex-1 py-2 bg-blue-100 text-blue-700 font-black text-center rounded-lg border border-blue-200">DER</span>
                            <span className="flex-1 py-2 bg-red-100 text-red-700 font-black text-center rounded-lg border border-red-200">DIE</span>
                            <span className="flex-1 py-2 bg-green-100 text-green-700 font-black text-center rounded-lg border border-green-200">DAS</span>
                        </div>
                    </div>
                </div>

                {/* Start Button */}
                <button
                    onClick={() => selectedGroup && onStart(selectedGroup)}
                    className="w-full py-6 bg-slate-900 text-white rounded-2xl font-black text-2xl hover:bg-black shadow-xl transition-all active:scale-95 flex items-center justify-center gap-3 group mt-4"
                >
                    BEGIN EXPERIMENT
                    <Icon.ArrowRight className="group-hover:translate-x-1 transition-transform" />
                </button>
            </div>
        </div>
    );
};