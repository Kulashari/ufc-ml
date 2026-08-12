import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { ChangeEvent, KeyboardEvent, MouseEvent } from "react";

import { searchFighters } from "../api";
import type { FighterOption } from "../types";

const SEARCH_DELAY_MS = 250;

interface FighterAutocompleteProps {
  id: string;
  name: string;
  label: string;
  placeholder: string;
  value: FighterOption | null;
  onChange: (fighter: FighterOption | null) => void;
  excludeFighterId?: string;
  disabled?: boolean;
  enterKeyHint?: "enter" | "done" | "go" | "next" | "previous" | "search" | "send";
}

function isAbortError(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "name" in error &&
    (error as { name: unknown }).name === "AbortError"
  );
}

export function FighterAutocomplete({
  id,
  name,
  label,
  placeholder,
  value,
  onChange,
  excludeFighterId,
  disabled = false,
  enterKeyHint,
}: FighterAutocompleteProps) {
  const generatedId = useId();
  const listboxId = `${id}-${generatedId}-options`;
  const statusId = `${id}-${generatedId}-status`;
  const [inputValue, setInputValue] = useState(value?.name ?? "");
  const [suggestions, setSuggestions] = useState<FighterOption[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [searchFailed, setSearchFailed] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const previousValueRef = useRef<FighterOption | null>(value);
  const optionRefs = useRef<Array<HTMLLIElement | null>>([]);

  const visibleSuggestions = useMemo(
    () => suggestions.filter((fighter) => fighter.id !== excludeFighterId),
    [excludeFighterId, suggestions],
  );

  useEffect(() => {
    const previousValue = previousValueRef.current;
    if (value) {
      setInputValue(value.name);
    } else if (previousValue) {
      setInputValue((current) => (current === previousValue.name ? "" : current));
    }
    previousValueRef.current = value;
  }, [value]);

  useEffect(() => {
    const query = inputValue.trim();
    const isSelectedValue = Boolean(value && inputValue === value.name);

    if (disabled || !query || isSelectedValue) {
      setSuggestions([]);
      setIsSearching(false);
      setHasSearched(false);
      setSearchFailed(false);
      setActiveIndex(-1);
      setIsOpen(false);
      return;
    }

    const controller = new AbortController();
    setIsSearching(true);
    setHasSearched(false);
    setSearchFailed(false);

    const timer = window.setTimeout(async () => {
      try {
        const fighters = await searchFighters(query, controller.signal);
        setSuggestions(fighters);
        setHasSearched(true);
        setIsOpen(true);
      } catch (error) {
        if (!isAbortError(error)) {
          setSuggestions([]);
          setHasSearched(true);
          setSearchFailed(true);
          setIsOpen(true);
        }
      } finally {
        if (!controller.signal.aborted) {
          setIsSearching(false);
        }
      }
    }, SEARCH_DELAY_MS);

    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [disabled, inputValue, value]);

  useEffect(() => {
    setActiveIndex((current) => {
      if (!visibleSuggestions.length) {
        return -1;
      }
      return current >= visibleSuggestions.length ? visibleSuggestions.length - 1 : current;
    });
  }, [visibleSuggestions.length]);

  useEffect(() => {
    if (activeIndex >= 0) {
      optionRefs.current[activeIndex]?.scrollIntoView({ block: "nearest" });
    }
  }, [activeIndex]);

  function selectFighter(fighter: FighterOption) {
    setInputValue(fighter.name);
    setSuggestions([]);
    setActiveIndex(-1);
    setIsOpen(false);
    setHasSearched(false);
    setSearchFailed(false);
    onChange(fighter);
  }

  function handleInputChange(event: ChangeEvent<HTMLInputElement>) {
    const nextValue = event.target.value;
    setInputValue(nextValue);
    setSuggestions([]);
    setActiveIndex(-1);
    setHasSearched(false);
    setSearchFailed(false);
    setIsOpen(Boolean(nextValue.trim()));
    if (value) {
      onChange(null);
    }
  }

  function handleKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "ArrowDown" && visibleSuggestions.length) {
      event.preventDefault();
      setIsOpen(true);
      setActiveIndex((current) => (current + 1) % visibleSuggestions.length);
      return;
    }

    if (event.key === "ArrowUp" && visibleSuggestions.length) {
      event.preventDefault();
      setIsOpen(true);
      setActiveIndex((current) =>
        current <= 0 ? visibleSuggestions.length - 1 : current - 1,
      );
      return;
    }

    if (event.key === "Enter" && isOpen && activeIndex >= 0) {
      const activeFighter = visibleSuggestions[activeIndex];
      if (activeFighter) {
        event.preventDefault();
        selectFighter(activeFighter);
      }
      return;
    }

    if (event.key === "Escape") {
      setIsOpen(false);
      setActiveIndex(-1);
    }
  }

  function keepInputFocused(event: MouseEvent<HTMLLIElement>) {
    event.preventDefault();
  }

  const showMenu = isOpen && Boolean(inputValue.trim()) && !value;
  const activeDescendant =
    showMenu && activeIndex >= 0 ? `${listboxId}-option-${activeIndex}` : undefined;
  const statusMessage = searchFailed
    ? "Fighter search is unavailable. Try again."
    : isSearching
      ? "Searching for fighters."
      : hasSearched
        ? `${visibleSuggestions.length} fighter suggestion${visibleSuggestions.length === 1 ? "" : "s"} available.`
        : "Start typing to search for a fighter.";

  return (
    <div className="fighter-autocomplete">
      <label className="field-label" htmlFor={id}>{label}</label>
      <div className="autocomplete-control">
        <input
          id={id}
          name={name}
          type="text"
          role="combobox"
          value={inputValue}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          onFocus={() => {
            if (inputValue.trim() && !value) {
              setIsOpen(true);
            }
          }}
          onBlur={() => {
            setIsOpen(false);
            setActiveIndex(-1);
          }}
          placeholder={placeholder}
          autoComplete="off"
          autoCapitalize="words"
          enterKeyHint={enterKeyHint}
          spellCheck={false}
          disabled={disabled}
          aria-autocomplete="list"
          aria-controls={showMenu ? listboxId : undefined}
          aria-describedby={statusId}
          aria-expanded={showMenu}
          aria-activedescendant={activeDescendant}
          required
        />
        {isSearching ? <span className="autocomplete-spinner spinner" aria-hidden="true" /> : null}

        {showMenu ? (
          <ul className="autocomplete-menu" id={listboxId} role="listbox">
            {visibleSuggestions.map((fighter, index) => (
              <li
                className={`autocomplete-option${index === activeIndex ? " is-active" : ""}`}
                id={`${listboxId}-option-${index}`}
                key={fighter.id}
                role="option"
                aria-selected={index === activeIndex}
                ref={(element) => {
                  optionRefs.current[index] = element;
                }}
                onMouseDown={keepInputFocused}
                onClick={() => selectFighter(fighter)}
              >
                {fighter.name}
              </li>
            ))}

            {isSearching && !visibleSuggestions.length ? (
              <li className="autocomplete-message" role="presentation">
                Searching fighters...
              </li>
            ) : null}
            {!isSearching && hasSearched && !searchFailed && !visibleSuggestions.length ? (
              <li className="autocomplete-message" role="presentation">
                No fighters found.
              </li>
            ) : null}
            {searchFailed ? (
              <li className="autocomplete-message autocomplete-error" role="presentation">
                Fighter search is unavailable. Try again.
              </li>
            ) : null}
          </ul>
        ) : null}
      </div>
      <span className="sr-only" id={statusId} aria-live="polite">{statusMessage}</span>
    </div>
  );
}
