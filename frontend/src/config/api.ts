// src/config/api.ts

/**
 * Sztywny adres backendu.
 * Używa zmiennych środowiskowych Vite (import.meta.env) do wykrycia trybu developera.
 */
export const API_BASE = import.meta.env.DEV
  ? 'http://127.0.0.1:8000' 
  : 'https://german-learning-language-backend.onrender.com';

// Logowanie dla celów debugowania (uruchomi się raz przy starcie aplikacji)
console.log("Current API Mode:", import.meta.env.DEV ? "Development (Local)" : "Production");
console.log("Connecting to:", API_BASE);