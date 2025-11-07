// src/api/predict.js
import api from './client'

export async function predictText(text) {
    const { data } = await api.post('/api/predict', { text })
    // { label: "Spam"|"Ham", probability: 0..1, confidence_pct: 0..100 }
    return data
}
