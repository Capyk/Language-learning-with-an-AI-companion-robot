import { useState, useEffect } from 'react';
import { Icon } from '../../../components/ui/Icons';

interface IntroScreenProps {
    onStart: (cond: 'A' | 'B') => void;
    assignedGroup: 'A' | 'B';
}

export const IntroScreen = ({ onStart, assignedGroup }: IntroScreenProps) => {
    // Stan zarządzający etapem: 'consent' (zgoda) lub 'instructions' (instrukcja)
    const [step, setStep] = useState<'consent' | 'instructions'>('consent');
    // REMOVED: const [selectedGroup, setSelectedGroup] = useState<'A' | 'B' | null>(null);
    // acceptedGroup comes from props

    // --- LOGIKA CONSENT FORM ---

    // --- LOGIKA CONSENT FORM ---
    const [showFullText, setShowFullText] = useState(false);
    const [consents, setConsents] = useState({
        isAdult: false,
        participate: false
    });

    const canProceed = consents.isAdult && consents.participate;

    const toggleConsent = (key: keyof typeof consents) => {
        setConsents(prev => ({ ...prev, [key]: !prev[key] }));
    };

    const handleConsentSubmit = () => {
        // REMOVED: setSelectedGroup(group);
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
                                    <li>Complete a learning section with vocabulary tasks</li>
                                    <li>Optionally interact with an AI tutor (limited use)</li>
                                    <li>Complete a short post-test and questionnaire</li>
                                </ul>
                                <p className="text-slate-500 text-sm mt-2">Duration: ~15 minutes</p>
                            </div>
                            <div>
                                <h3 className="font-bold text-lg mb-2">Data collected</h3>
                                <ul className="list-disc ml-5 text-slate-600 space-y-1">
                                    <li>Task answers and response times</li>
                                    <li>Feedback condition (static or AI-based)</li>
                                    <li>AI tutor interaction data (number of prompts used)</li>
                                    <li>Questionnaire responses</li>
                                    <li>Basic demographic information (Age, Gender, Education, Proficiency)</li>
                                </ul>
                            </div>
                        </div>

                        <div className="grid md:grid-cols-2 gap-6">
                            <div>
                                <h3 className="font-bold text-lg mb-2">AI Tutor</h3>
                                <ul className="list-disc ml-5 text-slate-600 space-y-1">
                                    <li>Some participants can use an AI-based tutor during learning</li>
                                    <li>The tutor provides short explanations and hints</li>
                                    <li>You may use up to 3 prompts maximum during the learning phase</li>
                                </ul>
                            </div>
                            <div>
                                <h3 className="font-bold text-lg mb-2">Your rights</h3>
                                <ul className="list-disc ml-5 text-slate-600 space-y-1">
                                    <li>Participation is voluntary</li>
                                    <li>You can stop at any time without consequences</li>
                                </ul>
                                <h3 className="font-bold text-lg mt-4 mb-2">Legal basis</h3>
                                <p className="text-slate-600">Data processing is based on your consent (GDPR Art. 6(1)(a))</p>
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
                            <h2 className="font-black text-lg text-slate-800 mb-2">Participant Information and Informed Consent (Full Version for the link)</h2>

                            <p className="mb-2"><strong>Study Title:</strong> Human–AI Interaction in Vocabulary Learning</p>
                            <p className="mb-2"><strong>Institution:</strong> Technical University of Munich (TUM)</p>
                            <p className="mb-2"><strong>Course:</strong> Human–AI Interaction</p>
                            <p className="mb-4"><strong>Researchers:</strong> Elham Tajalli, Alexander Brehmer, Ugur Alimoglu, Kacper Kolodziejczyk</p>
                            <p className="mb-4 text-xs">Contact: elham.tajalli@tum.de, alexander.brehmer@tum.de, ugur.alimoglu@tum.de, kacper.kolodziejczyk@tum.de</p>

                            <hr className="my-4 border-slate-100" />

                            <h3 className="font-bold text-slate-800 mt-4">1. Purpose of the Study</h3>
                            <p className="mb-2">You are invited to participate in a research study conducted as part of a university course project. The purpose of this study is to investigate how different types of feedback in a web-based vocabulary learning system influence learner engagement and short-term learning outcomes.</p>
                            <p className="mb-2">Specifically, the study compares:</p>
                            <ul className="list-disc ml-5 mb-2">
                                <li>static (non-adaptive) feedback, and</li>
                                <li>adaptive, AI-generated feedback.</li>
                            </ul>

                            <h3 className="font-bold text-slate-800 mt-4">2. Study Procedure</h3>
                            <p className="mb-2">If you agree to participate, you will complete the following steps:</p>
                            <ol className="list-decimal ml-5 mb-2">
                                <li>Read and agree to this consent form</li>
                                <li>Complete a short vocabulary pre-test</li>
                                <li>Complete a learning session involving image-based German vocabulary tasks</li>
                                <li>Optıonally interact with an AI tutor during learning</li>
                                <li>Complete a post-test and a short questionnare</li>
                            </ol>
                            <p className="mb-2">The study takes approximately 12–15 minutes in total.</p>

                            <h3 className="font-bold text-slate-800 mt-4">3. Tasks and Interaction</h3>
                            <p className="mb-2">During the learning phase, you will complete simple German vocabulary tasks (e.g., selecting the correct article for an image).</p>
                            <p className="mb-2">Some participants will have access to an AI-based tutor that can provide short explanations or hints related to the task.</p>
                            <p className="mb-2">Important details about the AI tutor:</p>
                            <ul className="list-disc ml-5 mb-2">
                                <li>Tutor interaction is optional</li>
                                <li>You may submit up to 3 prompts maximum during the learning phase</li>
                                <li>The AI tutor does not evaluate performance or assign scores</li>
                                <li>The tutor is designed only to support learning</li>
                            </ul>

                            <h3 className="font-bold text-slate-800 mt-4">4. AI System Use</h3>
                            <p className="mb-2">The AI tutor uses an AI language model running locally (via Ollama) to generate short feedback messages.</p>
                            <ul className="list-disc ml-5 mb-2">
                                <li>No personal identifying information is provided to the AI system</li>
                                <li>AI-generated content is used solely for educational feedback</li>
                                <li>AI processing is performed locally; no data is sent to external cloud services</li>
                            </ul>

                            <h3 className="font-bold text-slate-800 mt-4">5. Data Collected</h3>
                            <p className="mb-2">The following data may be collected:</p>
                            <ul className="list-disc ml-5 mb-2">
                                <li>Vocabulary task responses</li>
                                <li>Response times</li>
                                <li>Assigned feedback condition</li>
                                <li>AI tutor usage data (e.g., number of prompts used)</li>
                                <li>Questionnaire responses</li>
                                <li>Basic demographic information, including:
                                    <ul className="list-circle ml-5">
                                        <li>Age (in years or age range)</li>
                                        <li>Gender</li>
                                        <li>Highest level of education</li>
                                        <li>Self-reported German language proficiency</li>
                                    </ul>
                                </li>
                            </ul>
                            <p className="mb-2">The study does not intentionally collect sensitive personal data.</p>

                            <h3 className="font-bold text-slate-800 mt-4">6. Legal Basis for Data Processing (GDPR)</h3>
                            <p className="mb-2">Data processing in this study is based on informed consent in accordance with Article 6(1)(a) of the General Data Protection Regulation (GDPR).</p>
                            <p className="mb-2">You may withdraw your consent at any time without giving a reason. Withdrawal does not affect the lawfulness of data processing carried out prior to withdrawal.</p>

                            <h3 className="font-bold text-slate-800 mt-4">7. Data Storage, Anonymization, and Retention</h3>
                            <ul className="list-disc ml-5 mb-2">
                                <li>Data will be stored on the researchers’ devices and/or university systems with restricted access.</li>
                                <li>Participants are identified using a randomly generated participant ID.</li>
                                <li>Data will be analyzed and reported only in aggregated and anonymized form.</li>
                                <li>No identifying information will be included in publications or reports.</li>
                            </ul>
                            <p className="mb-2">Data will be retained until 31.03.2026, after which it will be deleted.</p>

                            <h3 className="font-bold text-slate-800 mt-4">8. Voluntary Participation and Withdrawal</h3>
                            <p className="mb-2">Participation in this study is entirely voluntary. You may stop participating at any time by closing the browser window or selecting the exit option. There are no negative consequences for withdrawing.</p>

                            <h3 className="font-bold text-slate-800 mt-4">9. Risks and Benefits</h3>
                            <p className="mb-2">There are no known risks beyond those associated with normal computer use. While you may benefit from practicing German vocabulary, there is no guaranteed personal benefit from participation.</p>

                            <h3 className="font-bold text-slate-800 mt-4">10. Contact Information</h3>
                            <p className="mb-2">If you have questions about the study or your data, you may contact:</p>
                            <p className="mb-2">Researchers: elham.tajalli@tum.de, alexander.brehmer@tum.de, ugur.alimoglu@tum.de, kacper.kolodziejczyk@tum.de</p>
                            <p className="mb-2">Course instructor or teaching staff: efe.bozkir@tum.de</p>

                            <h3 className="font-bold text-slate-800 mt-4">11. Consent Statement</h3>
                            <p className="mb-2">By agreeing to participate, you confirm that:</p>
                            <ul className="list-disc ml-5 mb-2">
                                <li>You are at least 18 years old</li>
                                <li>You have read and understood the information above</li>
                                <li>You voluntarily consent to participate in this study</li>
                            </ul>
                        </div>
                    )}

                    <hr className="border-slate-200" />

                    {/* CONSENT CHECKBOXES */}
                    <div className="space-y-4 px-4 hidden">
                        {/* 
                            NOTE: The user request asks for checkoxes:
                            ☐ I am at least 18 years old
                            ☐ I agree to participate in this study
                            but then only asks for buttons [I agree and continue] [I do not agree]
                            
                            Usually with explicit buttons like "I agree and continue", the checkboxes are implicit or part of the flow.
                            However, the prompt specifically listed the checkboxes in the text.
                            So I will KEEP them, but I will REMOVE the "webcam" checkbox as it wasn't in the new text.
                        */}
                    </div>

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
                    </div>

                    {/* ACTION BUTTONS - AUTOMATIC ASSIGNMENT */}
                    <div className="mt-8 flex gap-4">
                        <button
                            disabled={!canProceed}
                            onClick={() => handleConsentSubmit()}
                            className="flex-1 py-5 bg-blue-600 text-white rounded-2xl font-bold text-xl hover:bg-blue-700 shadow-xl transition-all active:scale-95 disabled:bg-slate-300 disabled:shadow-none disabled:cursor-not-allowed flex justify-center items-center gap-2"
                        >
                            {canProceed ? "I agree and continue" : "Completing Consent..."}
                            {canProceed && <Icon.ArrowRight />}
                        </button>

                        <button
                            onClick={() => {
                                if (window.confirm("Are you sure you want to decline?")) {
                                    window.close(); // Tries to close tab
                                    window.location.href = "about:blank"; // Fallback
                                }
                            }}
                            className="flex-1 py-5 bg-slate-100 text-slate-500 rounded-2xl font-bold text-xl hover:bg-slate-200 hover:text-slate-700 transition-all active:scale-95 flex justify-center items-center gap-2"
                        >
                            I do not agree
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
                        {assignedGroup === 'A'
                            ? "In the learning phase, you will follow a structured path to study the vocabulary items with standard exercises."
                            : "In the learning phase, an AI tutor will analyze your mistakes and generate personalized exercises to help you improve."}
                    </p>
                </div>

                {assignedGroup === 'B' && (
                    <div className="bg-purple-50 p-6 rounded-3xl border border-purple-100 animate-in fade-in slide-in-from-bottom-4">
                        <div className="flex items-center gap-3 mb-3">
                            <Icon.Sparkles className="text-purple-600" size={24} />
                            <h3 className="font-bold text-xl text-slate-800">Meet Your AI Tutor</h3>
                        </div>
                        <p className="text-slate-600 leading-relaxed mb-3">
                            During the learning phase, you will see an <strong>AI Tutor panel</strong> on the right.
                            You can ask questions, request hints, or get explanations for your mistakes.
                        </p>
                        <div className="bg-amber-50 border border-amber-200 rounded-xl p-3 mt-3">
                            <p className="text-amber-800 font-bold text-sm flex items-center gap-2">
                                <Icon.Info size={16} />
                                <span>Important: You have a limit of <strong>3 questions</strong> per session. Use them wisely!</span>
                            </p>
                        </div>
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
                    onClick={() => onStart(assignedGroup)}
                    className="w-full py-6 bg-slate-900 text-white rounded-2xl font-black text-2xl hover:bg-black shadow-xl transition-all active:scale-95 flex items-center justify-center gap-3 group mt-4"
                >
                    BEGIN EXPERIMENT
                    <Icon.ArrowRight className="group-hover:translate-x-1 transition-transform" />
                </button>
            </div>
        </div>
    );
};