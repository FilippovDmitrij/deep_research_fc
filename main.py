import openai
import json
import os
import httpx
import asyncio
import pandas as pd
from datasets import load_dataset
import time
from pydantic import BaseModel
from typing import Literal

# --- Конфигурация ---
# Убедитесь, что ваш API-ключ OpenAI установлен как переменная окружения
os.environ["OPENAI_API_KEY"] = "sk-K0DkLtbB_....."

# URL of our local adapter from docker-compose
SEARXNG_ADAPTER_URL = "http://localhost:8000/search"

SYSTEM_PROMPT = """
# ROLE AND GOAL
You are a "Deep Research Agent", an advanced AI assistant. Your primary function is to conduct thorough, multi-step research to provide comprehensive, accurate, and well-supported answers to user queries. You are not a simple search engine; you are an analytical expert who synthesizes information from various sources.

# CORE PRINCIPLE: QUERY TRANSFORMATION & DECONSTRUCTION
This is your most critical instruction. You MUST NOT search for the user's exact query verbatim. Instead, you must first perform a "Query Analysis" step in your internal monologue (thought process).

1.  **Deconstruct:** Break down the user's query into its fundamental components: key entities, underlying questions, and the ultimate information goal.
2.  **Brainstorm & Rephrase:** Generate a set of 2-3 diverse and specific search queries. These queries should explore different angles, use synonyms, and target potentially relevant adjacent topics. This strategy is essential for discovering the most relevant sources, even if they don't match the user's original phrasing.
3.  **Hypothesize:** For each query, have a brief hypothesis about what kind of information it might yield.

# RESEARCH PROCESS
1.  **Iterative Search:** Execute your generated search queries one by one using the `search_the_web` tool.
2.  **Analyze & Adapt:** After each search (Observation), analyze the results.
    - Did you find relevant information?
    - Do the results from one search suggest a new, better query?
    - If a query fails, do not give up. Analyze why it failed and try a different, more refined query. This is a crucial part of the research process.
3.  **Synthesize:** Do not rely on a single source. Gather information from multiple relevant pages. Look for consensus and note any conflicting information. Your goal is to build a complete picture.

# FINAL ANSWER FORMULATION
Once you have gathered sufficient information from multiple search iterations and are confident in your findings, synthesize all the information into a final, coherent answer. The final answer should be short, clear, and directly answer the user's original query."""

SUMMARIZATION_PROMPT = """
You are an expert analyst. Your task is to extract key information from the provided text that directly answers the original user query.
Focus on facts, figures, main arguments, and conclusions. Ignore irrelevant details, boilerplate text, advertisements, and fluff.
The result should be a concise yet comprehensive summary of only the most critical data on the topic.

CRITICAL!
NEVER PROVIDE THE FINAL ANSWER YOURSELF! YOUR JOB IS TO EXTRACT THE MOST RELEVANT INFORMATION SO THAT A FINAL ANSWER CAN BE FORMED BASED ON IT.

Original user query: "{user_query}"

Text to analyze:
---
{content}
---

Extracted Information:
"""

class EvaluationResponse(BaseModel):
    evaluation: Literal["ok", "wrong"]

# --- Новая функция для оценки ответа ---
async def evaluate_answer(question: str, golden_answer: str, agent_answer: str) -> EvaluationResponse | str:
    """
    Сравнивает ответ агента с "золотым" ответом, используя LLM и Pydantic для структурированного вывода.
    Возвращает объект EvaluationResponse или строку с ошибкой.
    """
    evaluator_client = openai.AsyncOpenAI()

    # Промпт обновлен, чтобы четко инструктировать модель следовать Pydantic-схеме.
    system_prompt = """
    You are a strict and precise evaluator. Your sole purpose is to compare a generated agent's answer against a ground-truth "golden" answer based on the original question.

    You must provide your evaluation in a structured JSON format:
    `evaluation`: Your final verdict, which must be either "ok" or "wrong".

    - If the agent's answer is factually correct and fully aligns with the golden answer, set `evaluation` to "ok".
    - If the agent's answer is incorrect, incomplete, or contains any factual errors compared to the golden answer, set `evaluation` to "wrong".
    """

    user_prompt = f"""
    Original Question: "{question}"
    ---
    Golden Answer (Ground Truth): "{golden_answer}"
    ---
    Agent's Answer to Evaluate: "{agent_answer}"
    """

    try:
        # Используем новый метод .parse(), который автоматически парсит ответ в Pydantic-модель
        completion = await evaluator_client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=EvaluationResponse,  # Передаем класс Pydantic напрямую
            temperature=0,
        )

        response_message = completion.choices[0].message

        if response_message.parsed:
            # Если парсинг удался, возвращаем готовый объект
            return response_message.parsed
        else:
            # Если модель отказалась отвечать в нужном формате
            print(f"⚠️ Evaluation refusal: {response_message.refusal}")
            return "eval_refusal"

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return "eval_error"


# --- Класс ReActAgent---
class ReActAgent:
    def __init__(self, model="gpt-4o-mini", max_iterations=5):
        self.client = openai.AsyncOpenAI()
        self.model = model
        self.max_iterations = max_iterations
        self.user_query = ""
        self.visited_urls = set()
        self.tools = {"search_the_web": self.search_the_web}
        self.tools_specs = [
            {
                "type": "function",
                "function": {
                    "name": "search_the_web",
                    "description": "Searches the internet for information. Can also extract and analyze the full text from web pages.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "A specific, targeted search query."},
                            "max_results": {"type": "integer",
                                            "description": "Maximum number of results to return (default is 3)."}
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

    async def _summarize_content(self, content_to_summarize: str) -> str:
        prompt = SUMMARIZATION_PROMPT.format(user_query=self.user_query, content=content_to_summarize)
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0.2,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Error during summarization: {e}")
            return "Failed to process text."

    async def search_the_web(self, query: str, max_results: int = 5) -> str:
        """
        Усовершенствованный поиск с адаптивной логикой:
        1. Пытается найти `max_results` *новых* URL.
        2. Если находит уже посещенные URL, удваивает количество запрашиваемых результатов и делает новую попытку.
        3. Если все результаты в первом запросе новые, обрабатывает только первые max_results.
        """
        print(f"🚀 Performing smart search for '{query}' with a target of {max_results} new results...")

        final_new_results = []
        # Множество для отслеживания URL, которые поисковик вернул в рамках *этого конкретного* вызова
        # чтобы не зацикливаться на одних и тех же ссылках
        urls_seen_in_this_search = set()

        current_request_size = max_results
        initial_request = True
        max_attempts = 3  # Защита от бесконечного цикла

        for attempt in range(max_attempts):
            if len(final_new_results) >= max_results:
                break

            print(f"   -> Attempt {attempt + 1}/{max_attempts}: Requesting {current_request_size} results...")

            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(SEARXNG_ADAPTER_URL, json={
                        "query": query, "max_results": current_request_size, "include_raw_content": True
                    }, timeout=30.0)
                    response.raise_for_status()
                    data = response.json()
                except httpx.RequestError as e:
                    print(f"❌ Connection error to SearXNG adapter: {e}")
                    return "Error: Could not connect to the search service."

            if not data.get("results"):
                # Если даже на увеличенный запрос нет результатов, выходим
                break

            # Фильтруем результаты: только новые и уникальные для этого вызова
            new_urls_found_this_attempt = []
            skipped_count = 0
            for res in data["results"]:
                url = res.get('url')
                # Проверяем, что URL есть, он не был посещен ранее глобально и не обрабатывался в этом поиске
                if url and url not in self.visited_urls and url not in urls_seen_in_this_search:
                    new_urls_found_this_attempt.append(res)
                    urls_seen_in_this_search.add(url)
                else:
                    skipped_count += 1

            if new_urls_found_this_attempt:
                final_new_results.extend(new_urls_found_this_attempt)

            # Логика принятия решений
            if initial_request and skipped_count == 0:
                print(
                    f"   -> All {len(final_new_results)} initial results are new. Processing first {max_results} as per rule.")
                final_new_results = final_new_results[:max_results]
                break  # Все результаты новые, прекращаем поиск
            elif skipped_count > 0:
                print(f"   -> Found {skipped_count} previously visited URLs. Doubling search size for next attempt.")
                current_request_size *= 2  # Удваиваем размер для следующего запроса

            initial_request = False

        if not final_new_results:
            return f"Search for '{query}' yielded no new results to analyze after {max_attempts} attempts."

        # Обрезаем до нужного количества, если набрали больше
        results_to_process = final_new_results[:max_results]

        # Продолжаем с саммаризацией, как и раньше
        results_to_summarize = [res for res in results_to_process if res.get("raw_content")]
        if results_to_summarize:
            tasks = [self._summarize_content(res["raw_content"]) for res in results_to_summarize]
            summarized_texts = await asyncio.gather(*tasks)
            for res, summary in zip(results_to_summarize, summarized_texts):
                res["summarized_content"] = summary

        # Форматируем итоговый результат и добавляем в `visited_urls`
        formatted_result = f"Search results for query: '{query}':\n\n"
        for i, result in enumerate(results_to_process, 1):
            url = result['url']
            # Добавляем в посещенные только те, которые мы реально обработали
            self.visited_urls.add(url)

            formatted_result += f'[{i}] {result["title"]}\n   URL: {url}\n'
            if "summarized_content" in result:
                formatted_result += f'   Summary: {result["summarized_content"]}\n\n'
            else:
                formatted_result += f'   Snippet: {result.get("content", "N/A")}\n\n'

        return formatted_result

    # *** ИЗМЕНЕНИЕ 1: Метод run теперь возвращает словарь {answer, iterations} ***
    async def run(self, user_query: str) -> dict:
        self.user_query = user_query
        self.visited_urls.clear()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": self.user_query}]
        print(f"\n🎯 New task: {self.user_query}")

        for iteration in range(self.max_iterations):
            current_iteration_count = iteration + 1
            print(f"\n=== Iteration {current_iteration_count} ===")
            try:
                response = await self.client.chat.completions.create(
                    model=self.model, messages=messages, tools=self.tools_specs, tool_choice="auto"
                )
                response_message = response.choices[0].message
                if not response_message.tool_calls:
                    final_answer = response_message.content
                    print(f"\n✅ Final Answer (tools no longer needed):\n{final_answer}")
                    return {"answer": final_answer, "iterations": current_iteration_count}

                messages.append(response_message)
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    query = function_args.get('query')
                    print(f"🤔 Thought: I need to find information. My query: '{query}'.")
                    function_response = await self.tools[function_name](**function_args)
                    print(
                        f"🧐 Observation: I have received processed data (showing snippet):\n---\n{function_response[:1000]}...\n---")
                    messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name,
                                     "content": function_response})

            except Exception as e:
                print(f"An error occurred during iteration {current_iteration_count}: {e}")
                await asyncio.sleep(5)
                return {"answer": f"Error during execution: {e}", "iterations": current_iteration_count}

        print("\nIteration limit reached. Generating final answer...")
        final_response = await self.client.chat.completions.create(model=self.model, messages=messages)
        final_answer = final_response.choices[0].message.content
        return {"answer": final_answer, "iterations": self.max_iterations}


# --- Основная функция для запуска бенчмарка ---
async def main():
    print("1. Загрузка бенчмарка 'vtllms/sealqa'...")
    dataset = load_dataset("vtllms/sealqa", name="seal_0", split="test")
    print(f"   ...Загружено {len(dataset)} вопросов.")

    NUM_QUESTIONS_TO_RUN = 2
    dataset_subset = dataset.select(range(len(dataset)))

    print(f"\n2. Запуск агента для {len(dataset_subset)} вопросов...")

    agent = ReActAgent()
    results_data = []

    start_time = time.time()

    for i, item in enumerate(dataset_subset):
        question = item['question']
        golden_answer = item['answer']
        agent_answer = "Not run"
        iterations_count = 0
        evaluation = "Not evaluated"  # Новая переменная для результата оценки

        print(f"\n--- [ {i + 1}/{len(dataset_subset)} ] Обработка вопроса: {question} ---")

        try:
            agent_result = await agent.run(question)
            agent_answer = agent_result.get("answer", "No answer provided")
            iterations_count = agent_result.get("iterations", 0)

            # *** НОВЫЙ ШАГ: Вызов оценщика ***
            print(f"⚖️ Evaluating the answer...")
            evaluation = await evaluate_answer(question, golden_answer, agent_answer)
        except Exception as e:
            print(f"Критическая ошибка при выполнении агента для вопроса: {question}\n{e}")

        results_data.append({
            'Вопрос': question,
            'Голден ответ': golden_answer,
            'Ответ от агента': agent_answer,
            'Количество итераций': iterations_count,
            'Оценка': evaluation  # *** НОВЫЙ ШАГ: Добавляем оценку в данные ***
        })
        print(f"--- [ {i + 1}/{len(dataset_subset)} ] Вопрос обработан за {iterations_count} итераций ---")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n3. Обработка завершена за {total_time:.2f} секунд.")

    print("\n4. Создание Excel отчета...")
    df = pd.DataFrame(results_data)
    # *** НОВЫЙ ШАГ: Добавляем новую колонку в отчет ***
    df = df[['Вопрос', 'Голден ответ', 'Ответ от агента', 'Оценка', 'Количество итераций']]
    output_filename = "benchmark_results_with_evaluation.xlsx"
    df.to_excel(output_filename, index=False, engine='openpyxl')

    print(f"   ...Отчет успешно сохранен в файл: {output_filename}")


if __name__ == "__main__":
    asyncio.run(main())
