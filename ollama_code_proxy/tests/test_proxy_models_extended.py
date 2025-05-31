import unittest
from ollama_code_proxy.proxy_server.models import (
    ModelTagInfo, TagsResponse, ModelDetails,
    ShowRequest, ShowResponse, ModelInformation,
    ChatMessage, ChatRequest, ChatResponse, ChatResponseMessage,
    PullRequest, PullStatus,
    DeleteRequest,
    CopyRequest,
    EmbedRequest, EmbedResponse,
    ProcessModelInfo, PsResponse,
    VersionResponse,
    CreateRequest,
    StatusOkResponse,
    OllamaRequest, OllamaResponse # Existing ones
)
from typing import List

class TestExtendedProxyModels(unittest.TestCase):

    def test_tags_response_instantiation(self):
        details = ModelDetails(parent_model="", format="gguf", family="llama", parameter_size="7B", quantization_level="Q4_0")
        tag_info = ModelTagInfo(name="llama2:latest", model="llama2:latest", modified_at="2023-10-01T10:00:00Z", size=4000000000, digest="sha256:abcdef", details=details)
        tags_response = TagsResponse(models=[tag_info])
        self.assertEqual(len(tags_response.models), 1)
        self.assertEqual(tags_response.models[0].name, "llama2:latest")

    def test_show_request_response_instantiation(self):
        show_req = ShowRequest(name="llama2:latest", verbose=True)
        self.assertEqual(show_req.name, "llama2:latest")

        # Test with alias population for ModelInformation
        model_info_data = {"general.architecture": "llama", "llama.block_count": 32, "general.file_type": 1}
        model_info = ModelInformation(**model_info_data) # type: ignore

        show_resp = ShowResponse(
            modelfile="FROM llama2...",
            parameters="num_ctx 2048",
            template="{{.Prompt}}",
            details=ModelDetails(parent_model="", format="gguf", family="llama", parameter_size="7B", quantization_level="Q4_0"),
            model_info=model_info,
            capabilities=["completion", "vision"]
        )
        self.assertTrue(show_resp.modelfile)
        self.assertEqual(show_resp.model_info.general_architecture, "llama") # type: ignore
        self.assertEqual(show_resp.model_info.general_file_type, 1) # type: ignore
        # Test extra field allowed by Config: extra = "allow"
        self.assertEqual(show_resp.model_info.model_extra["llama.block_count"], 32) # type: ignore


    def test_chat_request_response_instantiation(self):
        msg = ChatMessage(role="user", content="Hello")
        chat_req = ChatRequest(model="llama3", messages=[msg])
        self.assertEqual(chat_req.model, "llama3")
        self.assertEqual(chat_req.messages[0].role, "user")

        resp_msg = ChatResponseMessage(role="assistant", content="Hi there")
        chat_resp = ChatResponse(model="llama3", created_at="2023-10-01T10:00:00Z", message=resp_msg, done=True)
        self.assertTrue(chat_resp.done)
        self.assertEqual(chat_resp.message.role, "assistant") # type: ignore

    def test_pull_request_status_instantiation(self):
        pull_req = PullRequest(name="mistral:latest", stream=True)
        self.assertEqual(pull_req.name, "mistral:latest")
        pull_stat = PullStatus(status="downloading", digest="sha256:12345", total=1000, completed=50)
        self.assertEqual(pull_stat.status, "downloading")

    def test_delete_copy_request_instantiation(self):
        del_req = DeleteRequest(name="old-model")
        self.assertEqual(del_req.name, "old-model")
        copy_req = CopyRequest(source="model-a", destination="model-b")
        self.assertEqual(copy_req.source, "model-a")

    def test_embed_request_response_instantiation(self):
        embed_req_single = EmbedRequest(model="all-minilm", input="Some text")
        self.assertEqual(embed_req_single.model, "all-minilm")
        self.assertEqual(embed_req_single.input, "Some text")

        embed_req_list = EmbedRequest(model="all-minilm", input=["Some text", "Another text"])
        self.assertIsInstance(embed_req_list.input, List)
        self.assertEqual(len(embed_req_list.input), 2) # type: ignore

        embed_resp = EmbedResponse(embeddings=[[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(len(embed_resp.embeddings), 2)

    def test_ps_response_instantiation(self):
        details = ModelDetails(parent_model="", format="gguf", family="phi", parameter_size="3B", quantization_level="Q5_K_M")
        ps_model_info = ProcessModelInfo(
            name="phi3:latest", model="phi3:latest", modified_at="2023-10-01T12:00:00Z",
            size=3000000000, digest="sha256:fedcba", details=details,
            expires_at="2023-10-01T12:05:00Z", size_vram=3000000000
        )
        ps_resp = PsResponse(models=[ps_model_info])
        self.assertEqual(ps_resp.models[0].name, "phi3:latest")
        self.assertIsNotNone(ps_resp.models[0].expires_at)

    def test_version_response_instantiation(self):
        ver_resp = VersionResponse(version="0.1.32")
        self.assertEqual(ver_resp.version, "0.1.32")

    def test_create_request_instantiation(self):
        # Test with alias population for CreateRequest
        create_req = CreateRequest(name="my-custom-model", **{"from": "llama2"}) # type: ignore
        self.assertEqual(create_req.name, "my-custom-model")
        self.assertEqual(create_req.from_model, "llama2")

        create_req_direct = CreateRequest(name="my-custom-model-direct", from_model="llama3")
        self.assertEqual(create_req_direct.from_model, "llama3")


    def test_status_ok_response(self):
        ok_resp = StatusOkResponse()
        self.assertEqual(ok_resp.status, "success")
        ok_resp_custom = StatusOkResponse(status="deleted")
        self.assertEqual(ok_resp_custom.status, "deleted")

    def test_ollama_request_keep_alive_types(self):
        # Test original OllamaRequest with new keep_alive type
        req1 = OllamaRequest(model="m1", prompt="p", keep_alive="5m")
        self.assertEqual(req1.keep_alive, "5m")
        req2 = OllamaRequest(model="m2", prompt="p2", keep_alive=300)
        self.assertEqual(req2.keep_alive, 300)
        req3 = OllamaRequest(model="m3", prompt="p3") # keep_alive is Optional
        self.assertIsNone(req3.keep_alive)

    def test_ollama_request_optional_fields(self):
        # Test all optional fields in OllamaRequest
        req = OllamaRequest(
            model="test",
            prompt="test prompt",
            images=["img1"],
            format="json",
            options={"temp": 0.5},
            system="system prompt",
            template="template string",
            context=[1,2,3],
            stream=True,
            raw=True,
            keep_alive="-1",
            suffix="the end"
        )
        self.assertEqual(req.images, ["img1"])
        self.assertEqual(req.format, "json")
        self.assertEqual(req.options, {"temp": 0.5})
        self.assertEqual(req.system, "system prompt")
        self.assertEqual(req.template, "template string")
        self.assertEqual(req.context, [1,2,3])
        self.assertTrue(req.stream)
        self.assertTrue(req.raw)
        self.assertEqual(req.keep_alive, "-1")
        self.assertEqual(req.suffix, "the end")

    def test_ollama_response_optional_fields(self):
        resp = OllamaResponse(
            model="test",
            created_at="time",
            response="resp text",
            done=True,
            context=[1,2],
            total_duration=100,
            load_duration=10,
            prompt_eval_count=5,
            prompt_eval_duration=20,
            eval_count=30,
            eval_duration=70,
            done_reason="stop"
        )
        self.assertEqual(resp.context, [1,2])
        self.assertEqual(resp.total_duration, 100)
        self.assertEqual(resp.load_duration, 10)
        self.assertEqual(resp.prompt_eval_count, 5)
        self.assertEqual(resp.prompt_eval_duration, 20)
        self.assertEqual(resp.eval_count, 30)
        self.assertEqual(resp.eval_duration, 70)
        self.assertEqual(resp.done_reason, "stop")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
