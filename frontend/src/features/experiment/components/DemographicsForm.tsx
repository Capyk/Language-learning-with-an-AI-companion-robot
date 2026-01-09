import React, { useState, useEffect } from 'react';
import { Icon } from '../../../components/ui/Icons';

interface DemographicsFormProps {
    onSubmit: (data: any) => void;
}

export const DemographicsForm = ({ onSubmit }: DemographicsFormProps) => {
    const [formData, setFormData] = useState({ age: '', gender: '', education: '', german_level: '' });

    const handleChange = (key: string, val: string) => {
        setFormData(prev => ({ ...prev, [key]: val }));
    };

    const isComplete = formData.age && formData.gender && formData.education && formData.german_level;
    
    useEffect(() => { window.scrollTo(0,0); }, []);

    return (
        <div className="w-full max-w-2xl mx-auto my-12 animate-in fade-in slide-in-from-bottom-8 duration-700">
            
            <div className="bg-white border-t-8 border-green-600 rounded-xl shadow-sm p-8 mb-6 text-center">
                <h1 className="text-3xl font-black text-slate-900 mb-2">Almost Done!</h1>
                <p className="text-slate-600">Please provide some basic information about yourself to finalize the study.</p>
            </div>

            <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-8 space-y-8">
                
                {/* Age */}
                <div>
                    <label className="block text-slate-700 font-bold mb-2 text-lg">Age <span className="text-red-500">*</span></label>
                    <input 
                        type="number" 
                        min="10" max="99" 
                        value={formData.age}
                        onChange={e => handleChange('age', e.target.value)} 
                        // --- POPRAWIONO STYLE ---
                        className="w-full p-4 border border-slate-300 rounded-lg focus:ring-2 focus:ring-green-500 outline-none bg-white text-slate-900 font-medium transition-all text-lg placeholder-slate-400" 
                        placeholder="e.g. 24" 
                    />
                </div>

                {/* Gender */}
                <div>
                    <label className="block text-slate-700 font-bold mb-2 text-lg">Gender <span className="text-red-500">*</span></label>
                    <div className="grid grid-cols-2 gap-3">
                        {['Male', 'Female', 'Non-binary', 'Prefer not to say'].map(opt => (
                            <button 
                                key={opt}
                                type="button" 
                                onClick={() => handleChange('gender', opt)} 
                                className={`p-4 rounded-lg border-2 font-bold transition-all ${formData.gender === opt ? 'bg-green-600 text-white border-green-600' : 'bg-white text-slate-600 border-slate-200 hover:border-green-300'}`}
                            >
                                {opt}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Education */}
                <div>
                    <label className="block text-slate-700 font-bold mb-2 text-lg">Education Level <span className="text-red-500">*</span></label>
                    <select 
                        value={formData.education}
                        onChange={e => handleChange('education', e.target.value)} 
                        // --- POPRAWIONO STYLE ---
                        className="w-full p-4 border border-slate-300 rounded-lg focus:ring-2 focus:ring-green-500 outline-none bg-white text-slate-900 font-medium transition-all cursor-pointer text-lg"
                    >
                        <option value="" disabled className="text-slate-400">Select education...</option>
                        <option value="High School">High School</option>
                        <option value="Bachelor">Bachelor's</option>
                        <option value="Master">Master's</option>
                        <option value="PhD">PhD</option>
                        <option value="Other">Other</option>
                    </select>
                </div>

                {/* German Level */}
                <div>
                    <label className="block text-slate-700 font-bold mb-2 text-lg">German Proficiency <span className="text-red-500">*</span></label>
                    <div className="flex gap-2">
                        {['None', 'A1', 'A2', 'B1+'].map(lvl => (
                            <button 
                                key={lvl}
                                type="button"
                                onClick={() => handleChange('german_level', lvl)}
                                className={`flex-1 py-4 rounded-lg font-bold border-2 transition-all ${formData.german_level === lvl ? 'bg-slate-800 text-white border-slate-800' : 'bg-white text-slate-600 border-slate-200 hover:border-slate-400'}`}
                            >
                                {lvl}
                            </button>
                        ))}
                    </div>
                </div>

                <div className="pt-6">
                    <button 
                        disabled={!isComplete} 
                        onClick={() => onSubmit(formData)} 
                        className="w-full py-5 bg-green-600 text-white rounded-2xl font-black text-xl hover:bg-green-700 transition-all shadow-lg active:scale-95 disabled:bg-slate-300 disabled:shadow-none disabled:cursor-not-allowed flex justify-center items-center gap-3"
                    >
                        COMPLETE STUDY
                        <Icon.CheckCircle size={28} />
                    </button>
                </div>
            </div>
        </div>
    );
};