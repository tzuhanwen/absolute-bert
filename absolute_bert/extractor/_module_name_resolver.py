import re
from collections import defaultdict
from collections.abc import Iterable


class PrefixError(ValueError):
    pass


class IndexPositionError(ValueError):
    pass


class ModuleNameResolver:

    @classmethod
    def resolve_suffix_indices(
        cls, target: str, name_candidates: Iterable[str], recursive_root=True
    ) -> str:
        try:
            return cls._resolve_suffix_indices(target, name_candidates)

        except IndexError as e:
            if recursive_root:
                raise ValueError(  # noqa: B904
                    f"Can't use [index] syntax in module name: `{e.args[0]}`. "
                    "There may be some module having same prefix, "
                    "while not in the same module list."
                )
            raise

        except PrefixError as e:
            if recursive_root:
                raise ValueError(f"No module name with prefix: `{e.args[0]}`.")    # noqa: B904
            raise

    @classmethod
    def _resolve_suffix_indices(cls, target: str, name_candidates: Iterable[str]) -> str:

        # rule syntax: <prefix>[<index>]<remaining>
        # candidates: <prefix><suffix>, <suffix> = .<index><remaining>
        # remaining should start with `.`

        # catch patterns like `[N]`, `[-N].yyy.zz`
        index_remaining_pattern = r"\[(-?\d+)\](\..*)?"
        # first bracket and all afterwards
        m = re.search(index_remaining_pattern, target)
        if not m:
            return target

        first_bracket_position, _ = m.span()
        prefix = target[:first_bracket_position]  # string before first `[`
        index = int(m.group(1))  # e.g., -N
        remaining = m.group(2)  # e.g. yyy.zz in [-N].yyy.zz

        candidate_suffixes = [
            name[first_bracket_position:] for name in name_candidates if name.startswith(prefix)
        ]  # e.g., `.<index>`, `.<index>.yyy.zz`
        if not candidate_suffixes:
            raise PrefixError(prefix)

        # suffix must start with .<index> or be an empty string, e.g., `.2`, `.2.xxx`
        # index number will be in group1, and remaining part in group 2
        candidate_suffix_pattern = re.compile(r"(?:\.(\d+)(\..+)?$)")

        candidate_indices = set()
        candidate_remainings = defaultdict(set)  # index -> corresponding remainings
        for candidate_suffix in candidate_suffixes:

            suffix_re = candidate_suffix_pattern.match(candidate_suffix)
            if not suffix_re:
                # alignment failed
                raise IndexPositionError(prefix)

            candidate_index = suffix_re.group(1)
            candidate_remaining = suffix_re.group(2)

            # only count index only pattern, i.e., `.<index>`
            if candidate_remaining:
                candidate_remainings[candidate_index].add(candidate_remaining)
                continue

            if candidate_index is not None:
                candidate_indices.add(candidate_index)

        resolved_index = sorted(candidate_indices)[index]

        if remaining:
            try:
                resolved_remaining = cls.resolve_suffix_indices(
                    remaining, candidate_remainings[resolved_index], recursive_root=False
                )
                return f"{prefix}.{resolved_index}{resolved_remaining}"
            except IndexPositionError as e:
                raise IndexPositionError(f"{prefix}.{resolved_index}{e.args[0]}")  # noqa: B904
            except PrefixError as e:
                raise PrefixError(f"{prefix}.{resolved_index}{e.args[0]}")  # noqa: B904

        return f"{prefix}.{resolved_index}"
