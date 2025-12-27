import { Sparkles, Send, Lightbulb, FileQuestion, TrendingUp } from 'lucide-react';
import { useState } from 'react';

export function AIAssistant() {
  const [message, setMessage] = useState('');

  const conversationHistory = [
    {
      type: 'assistant',
      text: '¡Hola! Soy tu Asistente de IA especializado en bioestadística. Puedo ayudarte a interpretar resultados, elegir pruebas estadísticas adecuadas y explicar conceptos complejos. ¿En qué puedo ayudarte hoy?',
    },
    {
      type: 'user',
      text: '¿Cómo interpreto un p-valor de 0.023 en mi prueba t?',
    },
    {
      type: 'assistant',
      text: 'Un p-valor de 0.023 significa que hay un 2.3% de probabilidad de observar tus resultados (o más extremos) si la hipótesis nula fuera verdadera. Dado que es menor a 0.05 (nivel de significancia estándar), tienes evidencia estadísticamente significativa para rechazar la hipótesis nula. Esto sugiere que existe una diferencia significativa entre tus grupos.',
    },
  ];

  const quickActions = [
    {
      icon: FileQuestion,
      title: '¿Qué prueba usar?',
      description: 'Te ayudo a elegir la prueba estadística adecuada',
      color: 'from-purple-500 to-purple-600',
    },
    {
      icon: TrendingUp,
      title: 'Interpretar resultados',
      description: 'Explico tus análisis en lenguaje claro',
      color: 'from-blue-500 to-blue-600',
    },
    {
      icon: Lightbulb,
      title: 'Conceptos estadísticos',
      description: 'Aprende teoría de forma práctica',
      color: 'from-emerald-500 to-emerald-600',
    },
  ];

  const suggestedQuestions = [
    '¿Cuándo debo usar una prueba paramétrica vs no paramétrica?',
    'Explica el intervalo de confianza en mis resultados',
    '¿Qué tamaño de muestra necesito para 80% de potencia?',
    '¿Cómo reporto estos resultados en un paper?',
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <div className="w-14 h-14 bg-gradient-to-br from-purple-500 to-blue-600 rounded-2xl flex items-center justify-center">
          <Sparkles className="w-7 h-7 text-white" />
        </div>
        <div>
          <h1 className="text-2xl text-[#1e293b]">Asistente IA Estadístico</h1>
          <p className="text-gray-500">Potenciado por modelos avanzados de lenguaje</p>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {quickActions.map((action, index) => {
          const Icon = action.icon;
          return (
            <div 
              key={index}
              className={`bg-gradient-to-br ${action.color} rounded-2xl p-6 text-white cursor-pointer hover:shadow-xl transition-all`}
            >
              <Icon className="w-8 h-8 mb-3" />
              <h3 className="text-lg mb-2">{action.title}</h3>
              <p className="text-white/90 text-sm">{action.description}</p>
            </div>
          );
        })}
      </div>

      {/* Chat Interface */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
        {/* Conversation Area */}
        <div className="p-6 space-y-4 min-h-[400px] max-h-[500px] overflow-y-auto">
          {conversationHistory.map((msg, index) => (
            <div
              key={index}
              className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] rounded-2xl px-5 py-3 ${
                  msg.type === 'user'
                    ? 'bg-gradient-to-r from-[#2c5f7c] to-[#4a9ebb] text-white'
                    : 'bg-gray-100 text-gray-800'
                }`}
              >
                <p className="text-sm leading-relaxed">{msg.text}</p>
              </div>
            </div>
          ))}
        </div>

        {/* Input Area */}
        <div className="border-t border-gray-100 p-4 bg-gray-50">
          <div className="flex gap-3">
            <input
              type="text"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="Escribe tu pregunta sobre estadística..."
              className="flex-1 px-5 py-3 bg-white border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[#4a9ebb] focus:border-transparent"
            />
            <button className="px-6 py-3 bg-gradient-to-r from-[#2c5f7c] to-[#4a9ebb] text-white rounded-xl hover:shadow-lg transition-all flex items-center gap-2">
              <Send className="w-5 h-5" />
              Enviar
            </button>
          </div>
        </div>
      </div>

      {/* Suggested Questions */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <h3 className="text-lg text-[#1e293b] mb-4">Preguntas Sugeridas</h3>
        <div className="space-y-3">
          {suggestedQuestions.map((question, index) => (
            <button
              key={index}
              onClick={() => setMessage(question)}
              className="w-full text-left px-5 py-3 bg-gray-50 border border-gray-200 rounded-xl hover:border-[#4a9ebb] hover:bg-[#4a9ebb]/5 transition-all text-sm text-gray-700"
            >
              {question}
            </button>
          ))}
        </div>
      </div>

      {/* Tips */}
      <div className="bg-gradient-to-r from-purple-50 to-blue-50 border border-purple-200 rounded-2xl p-6">
        <div className="flex items-start gap-3">
          <Lightbulb className="w-6 h-6 text-purple-600 flex-shrink-0 mt-1" />
          <div>
            <h4 className="text-[#1e293b] mb-2">Consejo del Asistente</h4>
            <p className="text-sm text-gray-700 leading-relaxed">
              Para obtener las mejores respuestas, sé específico en tus preguntas. Incluye detalles como: 
              tipo de variables, tamaño de muestra, objetivo del análisis y contexto del estudio.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
