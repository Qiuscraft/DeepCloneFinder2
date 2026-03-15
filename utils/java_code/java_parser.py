from typing import List

import javalang
import javalang.tree
import javalang.tokenizer
import javalang.parse

from utils.data_class.function_info import FunctionInfo


class JavaParser:
    """Parse a Java source file and extract method information."""

    def __init__(self, file_path: str, source_code: str):
        self.file_path = file_path
        self.source_code = source_code

        self.lines = self.source_code.splitlines(True)
        self.tokens = list(javalang.tokenizer.tokenize(self.source_code))
        self.tree = self._parse_source_code()

    def _parse_source_code(self):
        return javalang.parse.parse(self.source_code)

    def _get_node_end_line(self, node) -> int:
        start_pos = node.position
        if not start_pos:
            return 0

        body_start_token_index = -1
        for i, token in enumerate(self.tokens):
            if token.position >= start_pos:
                if isinstance(token, javalang.tokenizer.Separator) and token.value == "{":
                    body_start_token_index = i
                    break

        if body_start_token_index == -1:
            for token in self.tokens:
                if token.position >= start_pos:
                    if isinstance(token, javalang.tokenizer.Separator) and token.value == ";":
                        if not token.position:
                            print(f"(error) [java_parser] not token.position 1, file_path: {self.file_path}")
                        else:
                            return token.position.line
            return start_pos.line

        open_braces = 0
        for token in self.tokens[body_start_token_index:]:
            if isinstance(token, javalang.tokenizer.Separator):
                if token.value == "{":
                    open_braces += 1
                elif token.value == "}":
                    open_braces -= 1
                    if open_braces == 0:
                        if not token.position:
                            print(f"(error) [java_parser] not token.position 2, file_path: {self.file_path}")
                        else:
                            return token.position.line

        return start_pos.line

    def extract_functions(self) -> List[FunctionInfo]:
        if not self.tree:
            return []

        functions: List[FunctionInfo] = []
        node_types = (javalang.tree.MethodDeclaration, javalang.tree.ConstructorDeclaration)

        for _, node in self.tree:
            if isinstance(node, node_types):
                if not node.position:
                    print(f"(error) [java_parser] not node.position, file_path: {self.file_path}")
                else:
                    start_line = node.position.line
                    end_line = self._get_node_end_line(node)
                    snippet_lines = self.lines[start_line - 1 : end_line]
                    code_snippet = "".join(snippet_lines).strip()
                    functions.append(
                        FunctionInfo(
                            id=-1,
                            start_line=start_line,
                            end_line=end_line,
                            code_snippet=code_snippet,
                            path=self.file_path,
                        )
                    )
        return functions
    