"""
LLM 服务 - 支持 DeepSeek 和本地模型
"""
import httpx
import json
from typing import Dict, Any, Optional, List
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import settings


class LLMService:
    """大语言模型服务"""
    
    def __init__(self, use_local: bool = False):
        """
        初始化 LLM 服务
        
        Args:
            use_local: 是否使用本地模型（内网部署的小模型）
        """
        self.use_local = use_local
        
        if use_local and settings.local_model_api_base:
            self.api_base = settings.local_model_api_base
            self.model = settings.local_model_name
            self.api_key = "dummy"  # 本地模型可能不需要 key
            logger.info(f"使用本地模型: {self.model}")
        else:
            self.api_base = settings.deepseek_api_base
            self.model = settings.deepseek_model
            self.api_key = settings.deepseek_api_key
            
            # 检查 API Key 是否配置
            if not self.api_key or self.api_key.strip() == "":
                logger.error("❌ DeepSeek API Key 未配置！")
                logger.error("请在项目根目录创建 .env 文件，添加：")
                logger.error("DEEPSEEK_API_KEY=your_api_key_here")
                raise ValueError("DeepSeek API Key 未配置，无法使用 LLM 服务")
            
            logger.info(f"✅ 使用 DeepSeek 模型: {self.model}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,  # 低温度保证稳定性
        max_tokens: int = 2000,
        response_format: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        调用 LLM API
        
        Args:
            messages: 对话消息列表
            temperature: 温度参数
            max_tokens: 最大 token 数
            response_format: 响应格式（如 {"type": "json_object"}）
        
        Returns:
            API 响应
        """
        import time
        start_time = time.time()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # DeepSeek 支持 JSON 模式
        if response_format:
            payload["response_format"] = response_format
        
        logger.info(f"🤖 调用 LLM API: {self.model}")
        logger.debug(f"   - Prompt 长度: {len(messages[-1]['content'])} 字符")
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # 提取使用信息
                    usage = result.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", 0)
                    
                    logger.info(f"✅ LLM 响应成功 (耗时: {elapsed:.2f}s)")
                    logger.info(f"   - Tokens: {prompt_tokens} (prompt) + {completion_tokens} (completion) = {total_tokens}")
                    logger.debug(f"   - 响应内容: {result['choices'][0]['message']['content'][:200]}...")
                    
                    return result
                else:
                    logger.error(f"❌ LLM API 错误: {response.status_code}")
                    logger.error(f"   - 响应: {response.text}")
                    raise Exception(f"LLM API 返回错误 {response.status_code}: {response.text}")
        
        except httpx.TimeoutException:
            elapsed = time.time() - start_time
            logger.error(f"❌ LLM API 请求超时 (已等待 {elapsed:.2f}s)")
            raise Exception("LLM API 请求超时，请检查网络连接")
        except httpx.ConnectError as e:
            logger.error(f"❌ 无法连接到 LLM API: {e}")
            raise Exception(f"无法连接到 API 服务器: {self.api_base}")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ LLM API 调用异常 (耗时: {elapsed:.2f}s): {e}")
            raise
    
    async def review_chunk(
        self,
        text: str,
        relevant_rules: List[Dict[str, Any]],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        审核文本块
        
        Args:
            text: 待审核文本
            relevant_rules: 相关规则
            context: 上下文信息
        
        Returns:
            审核结果（JSON 格式）
        """
        # 构造 prompt
        prompt = self._build_review_prompt(text, relevant_rules, context)
        
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的公文审核助手，负责检查文档是否符合写作标准。你必须严格按照标准进行审核，只标注明确违反标准的地方。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # 调用 LLM（要求 JSON 格式输出）
        response = await self.chat(
            messages=messages,
            temperature=0.1,  # 低温度保证一致性
            response_format={"type": "json_object"}  # DeepSeek 支持
        )
        
        # 解析响应
        content = response["choices"][0]["message"]["content"]
        
        try:
            result = json.loads(content)
            return result
        except json.JSONDecodeError:
            logger.error(f"LLM 返回的不是有效 JSON: {content}")
            # 尝试提取 JSON
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"issues": []}
    
    def _build_review_prompt(
        self,
        text: str,
        relevant_rules: List[Dict[str, Any]],
        context: Optional[str] = None
    ) -> str:
        """
        构造审核 prompt（优化版 - 更精简，适合小模型）
        
        优化点：
        1. 精简规则数量（只取最相关的1-2条）
        2. 简化示例（每条规则只给1个正例和1个反例）
        3. 使用更清晰的结构化格式
        4. 添加思维链引导
        """
        # 只取最相关的规则（减少 token 消耗）
        top_rules = relevant_rules[:2] if len(relevant_rules) > 2 else relevant_rules
        
        # 构建精简的 prompt
        prompt = f"""【审核任务】
检查以下文本是否违反写作标准。

【待审核文本】
{text}
"""
        
        # 添加上下文（如果有）
        if context:
            prompt += f"""
【上下文】
{context}
"""
        
        # 添加标准（精简版）
        prompt += "\n【适用标准】\n"
        for i, rule in enumerate(top_rules, 1):
            prompt += f"{i}. {rule.get('description', '')}\n"
            prompt += f"   规则ID: {rule.get('rule_id', '')} | 类别: {rule.get('category', '')} | 严重度: {rule.get('severity', 'medium')}\n"
            
            # 只给1个正例和1个反例
            positive_examples = rule.get('positive_examples', [])
            negative_examples = rule.get('negative_examples', [])
            
            if positive_examples:
                prompt += f"   ✅ 正确: {positive_examples[0]}\n"
            if negative_examples:
                prompt += f"   ❌ 错误: {negative_examples[0]}\n"
            prompt += "\n"
        
        # 添加思维链引导（帮助小模型更好地推理）
        prompt += """【审核步骤】
1. 逐句阅读文本
2. 对比每条标准
3. 找出明确违反的地方
4. 如果不确定，不要标注

【输出格式】JSON格式，示例：
{
  "issues": [
    {
      "position": "第X段",
      "rule_id": "R001",
      "category": "标题规范",
      "original_text": "原文片段（不超过30字）",
      "issue_description": "一句话说明问题",
      "suggestion": "修改建议",
      "confidence": 0.9
    }
  ]
}

如果没有问题，返回：{"issues": []}

现在开始审核，只返回JSON，不要其他内容。
"""
        
        return prompt

