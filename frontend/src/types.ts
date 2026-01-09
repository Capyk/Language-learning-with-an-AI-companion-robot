export interface TrialData {
    phase: string;
    index: number;
    total_in_phase: number;
    task_type: string;
    step_number?: number;
    title?: string;
    content?: string;
    image_url?: string;
    english_gloss?: string;
    german_word?: string;
    article?: string;
    plural?: string;
    example_sentence?: string;
    mnemonics?: string;
    question_context?: string;
    options?: string[];
    visual_type?: string;
    interaction_type?: string;
    payload?: any;
    status?: string;
}

export interface FeedbackData {
    is_correct: boolean;
    score: number;
    feedback: string;
    move_next?: boolean;
    transition?: boolean;
}