# from dataclasses import dataclass
# from jaxtyping import Array, Float
# from typing import List

# import jax.numpy as jnp


# @dataclass
# class CacheEntry:
#     past_keys: Float[Array, "... S NH HD"]
#     past_values: Float[Array, "... S NH DH"]

#     @classmethod
#     def init_cache_entry(cls, cfg, batch_dims: List[int] = []):
#         shape = (*tuple(batch_dims), 0, cfg.num_heads, cfg.head_dim)

#         return cls(
#             past_keys=jnp.empty(shape, dtype=cfg.dtype),
#             past_values=jnp.empty(shape, dtype=cfg.dtype),
#         )

#     def append(
#         self,
#         new_keys: Float[Array, "... NT NH HD"],
#         new_values: Float[Array, "... NT NH HD"],
#     ):
#         assert new_keys.shape[:-3] == self.past_keys.shape[:-3]
#         assert new_values.shape[:-3] == self.past_values.shape[:-3]

#         updated_keys: Float[Array, "... PNT NH HD"] = jnp.concatenate(
#             [self.past_keys, new_keys], axis=-3
#         )
#         updated_values: Float[Array, "... PNT NH HD"] = jnp.concatenate(
#             [self.past_values, new_values], axis=-3
#         )
#         self.past_keys = updated_keys
#         self.past_values = updated_values
#         return updated_keys, updated_values


# @dataclass
# class Cache:
#     entries: List[CacheEntry]

#     @classmethod
#     def init_cache(cls, cfg, batch_dims: List[int] = []):
#         return cls(
#             entries=[
#                 CacheEntry.init_cache_entry(cfg, batch_dims)
#                 for _ in range(cfg.num_layers)
#             ]
#         )

#     def __getitem__(self, idx) -> CacheEntry:
#         return self.entries[idx]

#     def __setitem__(self, idx, value):
#         self.entries[idx] = value
