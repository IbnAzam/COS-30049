// src/api/client.js
import axios from 'axios'

const api = axios.create({
  baseURL: '',   // stay relative; '/api/...' will hit the Vite proxy
  timeout: 10000,
})

export default api
